//! Detail tiles: everything that stands on the terrain of one map tile.
//! Buildings are grown from their OpenStreetMap footprint and height by a
//! rule set that looks at what is actually there - footprint shape, tagged
//! colour, the roof colour in the satellite image, land use, how dense and how
//! green the neighbourhood is - and picks a building type, roof form, facade
//! and level of upkeep from it. Roads, trees and parked cars are added by
//! their own modules.
use crate::geo::{self, TileId};
use crate::render::{RoadVertex, TreeInst};
use crate::sources::{mip_chain, ElevTile, Sources, VecTile, DATA_ZOOM, LU_COMMERCIAL, LU_INDUSTRIAL, LU_OPEN, LU_RESIDENTIAL};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec2, Vec3};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    /// walls: (metres along, metres above ground). roofs: material coordinates.
    pub uv: [f32; 2],
    /// rgb colour; a = surface kind (see K_*)
    pub col: [u8; 4],
    /// seed, facade type or roof material, upkeep / wealth, height in metres
    pub sty: [u8; 4],
}
pub const K_WALL: u8 = 0;
pub const K_FLAT: u8 = 50;
pub const K_PITCH: u8 = 100;
pub const K_PLAIN: u8 = 150;
pub const K_DECK: u8 = 200;
/// lamp heads and signal faces: glow (sty[1]: 0 street light, 1 traffic signal)
pub const K_LAMP: u8 = 250;

pub struct Footprint {
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub ring: Vec<[f32; 2]>,
    pub base: f32,
    pub top: f32,
    /// 0 building, 1 bridge deck
    pub kind: u8,
    pub col: [u8; 3],
    /// index ranges (start, count) in the tile's main, detail and fine-detail index runs
    pub main: (u32, u32),
    pub det: (u32, u32),
    pub fine: (u32, u32),
    /// index range of its far stand-in
    pub far: (u32, u32),
    pub alive: bool,
    /// centre of the footprint at ground level, relative to the tile centre
    pub center: Vec3,
    pub area: f32,
    /// the walls' style (seed, facade type, upkeep, height) and the altitude
    /// their facade coordinates count up from, for drawing the wreck the same way
    pub sty: [u8; 4],
    pub gref: f32,
    pub roof_col: [u8; 3],
    /// setbacks near the top of a tower: (inset from the ring below, top of that tier)
    pub tiers: Vec<(f32, f32)>,
}

pub struct TreeChunk {
    pub center: Vec3,
    pub radius: f32,
    /// (variant, first instance, count)
    pub runs: Vec<(u16, u32, u32)>,
}

pub struct Parked {
    pub pos: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
    pub model: u8,
    pub col: [f32; 3],
}

pub struct BuildingData {
    pub id: TileId,
    pub center: DVec3,
    pub radius: f64,
    pub ext: f32,
    pub unit_m: f32,
    pub hoods: Hoods,
    pub landuse: Vec<u8>,
    pub verts: Vec<BVertex>,
    pub indices: Vec<u32>,
    /// indices without the small rooftop detail
    pub n_main: u32,
    /// end of the middle level of detail; the finest level runs from here to `n_far`
    pub n_det: u32,
    /// end of the buildings' own part of the base level; the rest of it
    /// (bridge decks) is drawn at every distance
    pub n_bld: u32,
    /// start of the far stand-ins (they run to the end), and end of those of
    /// buildings tall enough to matter in the far shadow cascade
    pub n_far: u32,
    pub n_far_tall: u32,
    /// Where each height class (see `HCLASS`) ends in the building part of the
    /// base level, the middle level, the finest level and the far stand-ins
    /// (absolute indices; buildings are laid down tallest first in each).
    pub cls: [[u32; 4]; 4],
    pub atlas: Vec<Vec<u8>>,
    pub atlas_size: u32,
    pub ax: [f32; 4],
    pub ay: [f32; 4],
    pub footprints: Vec<Footprint>,
    pub road_verts: Vec<RoadVertex>,
    pub road_idx: Vec<u32>,
    pub net: crate::roads::RoadNet,
    pub parked: Vec<Parked>,
    pub trees: Vec<TreeInst>,
    pub tree_var: Vec<u8>,
    pub chunks: Vec<TreeChunk>,
    /// forest-floor plants, sorted by cell
    pub under: Vec<crate::trees::Under>,
    pub cells: Vec<crate::trees::NearCell>,
    /// tree indices sorted by cell
    pub near_idx: Vec<u32>,
}

pub const ATLAS: usize = 512;
pub const RASTER: usize = 1024;

pub fn point_in_ring(p: [f32; 2], r: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = r.len() - 1;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub fn hash(a: u32, b: u32) -> f32 {
    let mut h = a.wrapping_mul(0x9e3779b1) ^ b.wrapping_mul(0x85ebca6b).rotate_left(13);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a2d39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Region {
    NAm,
    LatAm,
    NEur,
    Med,
    Mena,
    Africa,
    SAsia,
    EAsia,
    SeAsia,
    Oceania,
}
/// Very coarse cultural region from degrees; enough to pick building habits.
pub fn region(lat: f64, lon: f64) -> Region {
    if lon < -30.0 {
        if lat > 30.0 { Region::NAm } else { Region::LatAm }
    } else if lon < 60.0 {
        if lat > 46.0 {
            Region::NEur
        } else if lat > 35.0 && lon < 45.0 {
            Region::Med
        } else if lat > 14.0 {
            Region::Mena
        } else {
            Region::Africa
        }
    } else if lat < -9.0 && lon > 110.0 {
        Region::Oceania
    } else if lat > 50.0 {
        Region::NEur
    } else if lon < 93.0 {
        if lat > 36.0 { Region::Mena } else { Region::SAsia }
    } else if lat < 24.0 {
        Region::SeAsia
    } else {
        Region::EAsia
    }
}

/// Shared geometry and imagery context for one detail tile.
pub struct Ctx {
    pub id: TileId,
    pub ext: f32,
    pub unit_m: f32,
    pub center: DVec3,
    pub elev: Option<Arc<ElevTile>>,
    pub atlas: Vec<u8>,
    pub lat: f64,
    pub lon: f64,
    pub region: Region,
}
impl Ctx {
    pub fn ground(&self, p: [f32; 2]) -> f32 {
        self.elev.as_ref().map(|e| e.sample((p[0] / self.ext) as f64, (p[1] / self.ext) as f64)).unwrap_or(0.0)
    }
    pub fn pos(&self, p: [f32; 2], alt: f32) -> Vec3 {
        let (lat, lon) = self.id.lat_lon((p[0] / self.ext) as f64, (p[1] / self.ext) as f64);
        (geo::lla_to_pos(lat, lon, alt as f64) - self.center).as_vec3()
    }
    /// Metres (east, south) from the tile centre -> tile units.
    pub fn units(&self, m: Vec2) -> [f32; 2] {
        [m.x / self.unit_m + self.ext * 0.5, m.y / self.unit_m + self.ext * 0.5]
    }
    pub fn metres(&self, p: [f32; 2]) -> Vec2 {
        Vec2::new((p[0] - self.ext * 0.5) * self.unit_m, (p[1] - self.ext * 0.5) * self.unit_m)
    }
    pub fn pos_m(&self, m: Vec2, alt: f32) -> Vec3 {
        self.pos(self.units(m), alt)
    }
    /// Satellite colour (sRGB bytes) at a tile position.
    pub fn img(&self, p: [f32; 2]) -> [u8; 3] {
        let x = ((p[0] / self.ext) * ATLAS as f32).clamp(0.0, ATLAS as f32 - 1.0) as usize;
        let y = ((p[1] / self.ext) * ATLAS as f32).clamp(0.0, ATLAS as f32 - 1.0) as usize;
        let px = &self.atlas[(y * ATLAS + x) * 4..];
        [px[0], px[1], px[2]]
    }
}

/// How vegetated a satellite pixel looks, 0..1: green-dominant and not bright.
pub fn greenness(px: [u8; 3]) -> f32 {
    let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
    let sum = r + g + b + 1.0;
    let exg = (2.0 * g - r - b) / sum;
    let lum = sum / 3.0;
    ((exg - 0.015) / 0.09).clamp(0.0, 1.0) * ((175.0 - lum) / 70.0).clamp(0.0, 1.0)
}

/// Level of detail a piece of geometry belongs to: 0 always drawn, 1 middle
/// distance (parapets, chimneys, street furniture), 2 close up only.
#[derive(Copy, Clone, PartialEq)]
pub struct Lvl(pub u8);
impl From<bool> for Lvl {
    fn from(det: bool) -> Self {
        Lvl(det as u8)
    }
}
pub const FINE: Lvl = Lvl(2);
/// The simplified stand-in for a building far away (see `far_version`).
pub const FAR: Lvl = Lvl(3);
/// Distances (from the camera) over which buildings dissolve from their full
/// geometry into their far stand-ins. Must match FAR0/FAR1 in buildings.wgsl.
pub const BFAR: [f32; 2] = [1000.0, 1300.0];
/// Distance the finest level of detail starts to dissolve out at (60% of
/// `LOD2_FAR` in buildings.wgsl; must match it).
pub const FINE_FADE: f32 = 680.0 * 0.6;
/// Height classes, tallest first: (least height in metres, distance scale). A
/// tower stands out against the skyline from much farther off than a house, so
/// its full geometry, and the middle and finest levels that dress it, are kept
/// out to `scale` times the usual distances (BFAR, LOD2_FAR in buildings.wgsl
/// and the tile reaches in world.rs). About height / 50 m: what a 50 m block
/// shows at a kilometre, a 150 m tower shows at three.
pub const HCLASS: [(f32, f32); 4] = [(150.0, 3.0), (100.0, 2.0), (65.0, 1.5), (0.0, 1.0)];
pub fn hclass(h: f32) -> usize {
    HCLASS.iter().position(|c| h >= c.0).unwrap_or(HCLASS.len() - 1)
}
/// Most vertices a tile may spend on its finest level of detail.
const FINE_BUDGET: usize = 450_000;

#[derive(Default)]
pub struct Mesh {
    pub v: Vec<BVertex>,
    pub main: Vec<u32>,
    pub det: Vec<u32>,
    pub fine: Vec<u32>,
    pub far: Vec<u32>,
    pub radius: f32,
    fine_verts: usize,
}
impl Mesh {
    /// Whether there is still room for close-up detail in this tile.
    pub fn fine_ok(&self) -> bool {
        self.fine_verts < FINE_BUDGET
    }
    /// Convex polygon as a fan. `up` (if given) forces the normal to that side.
    pub fn poly(&mut self, lvl: impl Into<Lvl>, pts: &[(Vec3, [f32; 2])], col: [u8; 4], sty: [u8; 4], up: Option<Vec3>) {
        let lvl = lvl.into();
        if pts.len() < 3 {
            return;
        }
        let mut n = Vec3::ZERO;
        for k in 1..pts.len() - 1 {
            n += (pts[k].0 - pts[0].0).cross(pts[k + 1].0 - pts[0].0);
        }
        let mut n = n.normalize_or_zero();
        if let Some(u) = up {
            if n.dot(u) < 0.0 {
                n = -n;
            }
        }
        let v0 = self.v.len() as u32;
        for (p, uv) in pts {
            self.radius = self.radius.max(p.length());
            self.v.push(BVertex { pos: (*p).into(), nrm: n.into(), uv: *uv, col, sty });
        }
        if lvl == FINE {
            self.fine_verts += pts.len();
        }
        let ix = match lvl.0 {
            0 => &mut self.main,
            1 => &mut self.det,
            2 => &mut self.fine,
            _ => &mut self.far,
        };
        for k in 1..pts.len() as u32 - 1 {
            ix.extend([v0, v0 + k, v0 + k + 1]);
        }
    }
    /// Box standing on a rectangle given by centre, two half-axes (metres, in the
    /// tile's metric frame) between two altitudes. No bottom face.
    #[allow(clippy::too_many_arguments)]
    pub fn boxm(&mut self, cx: &Ctx, lvl: impl Into<Lvl>, c: Vec2, a: Vec2, b: Vec2, z0: f32, z1: f32, col: [u8; 4], sty: [u8; 4], top: Option<[u8; 4]>) {
        let lvl = lvl.into();
        let cn = [c - a - b, c + a - b, c + a + b, c - a + b];
        let lo: Vec<Vec3> = cn.iter().map(|p| cx.pos_m(*p, z0)).collect();
        let hi: Vec<Vec3> = cn.iter().map(|p| cx.pos_m(*p, z1)).collect();
        let (la, lb) = (a.length() * 2.0, b.length() * 2.0);
        for k in 0..4 {
            let j = (k + 1) % 4;
            let l = if k % 2 == 0 { la } else { lb };
            self.poly(lvl, &[(lo[k], [0.0, 0.0]), (lo[j], [l, 0.0]), (hi[j], [l, z1 - z0]), (hi[k], [0.0, z1 - z0])], col, sty, None);
        }
        let upv = (hi[0] + cx.center.as_vec3()).normalize();
        self.poly(lvl, &[(hi[0], [0.0, 0.0]), (hi[1], [la, 0.0]), (hi[2], [la, lb]), (hi[3], [0.0, lb])], top.unwrap_or(col), sty, Some(upv));
    }
    /// Upright prism with `n` sides (a tank, a drum) standing on a circle.
    #[allow(clippy::too_many_arguments)]
    pub fn prism(&mut self, cx: &Ctx, lvl: impl Into<Lvl>, c: Vec2, r: f32, n: usize, z0: f32, z1: f32, col: [u8; 4], sty: [u8; 4], top: Option<[u8; 4]>) {
        let lvl = lvl.into();
        let pt = |k: usize| c + Vec2::from_angle(k as f32 / n as f32 * std::f32::consts::TAU) * r;
        let side = std::f32::consts::TAU * r / n as f32;
        for k in 0..n {
            let (a, b) = (pt(k), pt(k + 1));
            let (s0, s1) = (k as f32 * side, (k + 1) as f32 * side);
            self.poly(lvl, &[(cx.pos_m(a, z0), [s0, 0.0]), (cx.pos_m(b, z0), [s1, 0.0]), (cx.pos_m(b, z1), [s1, z1 - z0]), (cx.pos_m(a, z1), [s0, z1 - z0])], col, sty, None);
        }
        let upv = (cx.pos_m(c, z1) + cx.center.as_vec3()).normalize();
        let cap: Vec<(Vec3, [f32; 2])> = (0..n).map(|k| (cx.pos_m(pt(k), z1), [(pt(k) - c).x, (pt(k) - c).y])).collect();
        self.poly(lvl, &cap, top.unwrap_or(col), sty, Some(upv));
    }
}

// ------------------------------------------------------------------ roofs

#[derive(Copy, Clone, Debug)]
enum Roof {
    Flat,
    /// ridge height, ridge offset across the building
    Gable(f32, f32),
    /// pitch (rise per metre)
    Hip(f32),
    Shed(f32),
    /// steep pitch, flat-top height
    Mansard(f32, f32),
    /// bays, height: a row of parallel ridges
    Multi(u32, f32),
}
impl Roof {
    fn h(&self, u: f32, v: f32, hl: f32, hw: f32) -> f32 {
        self.h_raw(u, v, hl, hw).max(0.0)
    }
    /// Height of the roof plane, continued past the walls (for the eaves).
    fn h_raw(&self, u: f32, v: f32, hl: f32, hw: f32) -> f32 {
        match *self {
            Roof::Flat => 0.0,
            Roof::Gable(r, off) => {
                if v >= off { r * (hw - v) / (hw - off) } else { r * (hw + v) / (hw + off) }
            }
            Roof::Hip(p) => p * (hw - v.abs()).min(hl - u.abs()),
            Roof::Shed(p) => p * (v + hw),
            Roof::Mansard(p, cap) => (p * (hw - v.abs()).min(hl - u.abs())).min(cap),
            Roof::Multi(bays, r) => {
                let t = ((v + hw) / (2.0 * hw) * bays as f32).rem_euclid(1.0);
                r * (1.0 - (2.0 * t - 1.0).abs())
            }
        }
    }
    /// Lines a*u + b*v = c along which the roof surface folds.
    fn creases(&self, hl: f32, hw: f32) -> Vec<(f32, f32, f32)> {
        let d = hl - hw;
        match *self {
            Roof::Flat | Roof::Shed(_) => vec![],
            Roof::Gable(_, off) => vec![(0.0, 1.0, off)],
            Roof::Hip(_) => vec![(0.0, 1.0, 0.0), (1.0, -1.0, d), (1.0, -1.0, -d), (1.0, 1.0, d), (1.0, 1.0, -d)],
            Roof::Mansard(p, cap) => {
                let c = cap / p;
                vec![(1.0, -1.0, d), (1.0, -1.0, -d), (1.0, 1.0, d), (1.0, 1.0, -d), (0.0, 1.0, hw - c), (0.0, 1.0, c - hw), (1.0, 0.0, hl - c), (1.0, 0.0, c - hl)]
            }
            Roof::Multi(bays, _) => (1..2 * bays).map(|i| (0.0, 1.0, -hw + i as f32 * hw / bays as f32)).collect(),
        }
    }
    fn max_h(&self, hl: f32, hw: f32) -> f32 {
        match *self {
            Roof::Flat => 0.0,
            Roof::Gable(r, _) | Roof::Multi(_, r) => r,
            Roof::Hip(p) => p * hw.min(hl),
            Roof::Shed(p) => p * 2.0 * hw,
            Roof::Mansard(_, cap) => cap,
        }
    }
}

fn split_tris(tris: Vec<[Vec2; 3]>, line: (f32, f32, f32)) -> Vec<[Vec2; 3]> {
    let mut out = Vec::with_capacity(tris.len() + 8);
    let eps = 1e-3;
    for t in tris {
        let s = [line.0 * t[0].x + line.1 * t[0].y - line.2, line.0 * t[1].x + line.1 * t[1].y - line.2, line.0 * t[2].x + line.1 * t[2].y - line.2];
        if s.iter().all(|v| *v >= -eps) || s.iter().all(|v| *v <= eps) {
            out.push(t);
            continue;
        }
        // The vertex alone on its side of the line.
        let i0 = (0..3).find(|&i| s[i].abs() > eps && s[(i + 1) % 3] * s[i] <= 0.0 && s[(i + 2) % 3] * s[i] <= 0.0).unwrap_or(0);
        let (i1, i2) = ((i0 + 1) % 3, (i0 + 2) % 3);
        let cut = |a: usize, b: usize| t[a] + (t[b] - t[a]) * (s[a] / (s[a] - s[b])).clamp(0.0, 1.0);
        let (p01, p02) = (cut(i0, i1), cut(i0, i2));
        for n in [[t[i0], p01, p02], [p01, t[i1], t[i2]], [p01, t[i2], p02]] {
            if (n[1] - n[0]).perp_dot(n[2] - n[0]).abs() > 1e-4 {
                out.push(n);
            }
        }
    }
    out
}

/// Smallest enclosing rectangle: centre, unit long axis, half length, half width.
fn obb(pts: &[Vec2]) -> (Vec2, Vec2, f32, f32) {
    let mut best = (f32::MAX, Vec2::ZERO, Vec2::X, 1.0, 1.0);
    let n = pts.len();
    for i in 0..n {
        let e = pts[(i + 1) % n] - pts[i];
        if e.length() < 0.5 {
            continue;
        }
        let d = e.normalize();
        let p = d.perp();
        let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for q in pts {
            let (a, b) = (q.dot(d), q.dot(p));
            a0 = a0.min(a);
            a1 = a1.max(a);
            b0 = b0.min(b);
            b1 = b1.max(b);
        }
        let area = (a1 - a0) * (b1 - b0);
        if area < best.0 {
            let c = d * (a0 + a1) * 0.5 + p * (b0 + b1) * 0.5;
            best = if a1 - a0 >= b1 - b0 { (area, c, d, (a1 - a0) * 0.5, (b1 - b0) * 0.5) } else { (area, c, p, (b1 - b0) * 0.5, (a1 - a0) * 0.5) };
        }
    }
    (best.1, best.2, best.3.max(0.5), best.4.max(0.5))
}

/// Move every edge of a ring inward by `d`. None if the shape does not survive it.
pub fn inset(ring: &[Vec2], d: f32) -> Option<Vec<Vec2>> {
    let n = ring.len();
    if n < 3 {
        return None;
    }
    let area: f32 = (0..n).map(|i| ring[i].perp_dot(ring[(i + 1) % n])).sum::<f32>() * 0.5;
    let sign = area.signum();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (p, c, q) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
        let (e0, e1) = ((c - p).normalize_or_zero(), (q - c).normalize_or_zero());
        if e0 == Vec2::ZERO || e1 == Vec2::ZERO {
            return None;
        }
        // Inward normals (left of travel for positive area).
        let (n0, n1) = (e0.perp() * sign, e1.perp() * sign);
        let m = n0 + n1;
        let k = m.length_squared();
        if k < 0.15 {
            return None;
        }
        out.push(c + m * (2.0 * d / k));
    }
    let a2: f32 = (0..n).map(|i| out[i].perp_dot(out[(i + 1) % n])).sum::<f32>() * 0.5;
    if a2 * sign <= area.abs() * 0.08 {
        return None;
    }
    for i in 0..n {
        if (out[(i + 1) % n] - out[i]).dot(ring[(i + 1) % n] - ring[i]) <= 0.0 {
            return None;
        }
    }
    Some(out)
}

fn in_ring_m(p: Vec2, r: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = r.len() - 1;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ----------------------------------------------------------- neighbourhoods

const NB: usize = 8;
/// What the surroundings of a spot in the tile are like.
#[derive(Copy, Clone, Default, Debug)]
pub struct Hood {
    /// fraction of the ground covered by buildings
    pub cover: f32,
    /// fraction of the satellite image that is vegetation
    pub green: f32,
    /// 0 struggling .. 1 affluent
    pub wealth: f32,
    pub mean_h: f32,
}
#[derive(Clone, Default)]
pub struct Hoods(pub Vec<Hood>);
impl Hoods {
    pub fn at(&self, cx: &Ctx, p: [f32; 2]) -> Hood {
        self.sample_uv(p[0] / cx.ext, p[1] / cx.ext)
    }
    /// Bilinearly sample neighbourhood metrics from normalized tile coordinates `u, v` in `0..1`.
    pub fn sample_uv(&self, u: f32, v: f32) -> Hood {
        if self.0.len() < NB * NB {
            return Hood::default();
        }
        let f = |t: f32| (t * NB as f32 - 0.5).clamp(0.0, NB as f32 - 1.001);
        let (x, y) = (f(u), f(v));
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let g = |i: usize, j: usize| self.0[j.min(NB - 1) * NB + i.min(NB - 1)];
        let mix = |a: Hood, b: Hood, t: f32| Hood { cover: a.cover + (b.cover - a.cover) * t, green: a.green + (b.green - a.green) * t, wealth: a.wealth + (b.wealth - a.wealth) * t, mean_h: a.mean_h + (b.mean_h - a.mean_h) * t };
        mix(mix(g(x0, y0), g(x0 + 1, y0), fx), mix(g(x0, y0 + 1), g(x0 + 1, y0 + 1), fx), fy)
    }
    pub fn mean_cover(&self) -> f32 {
        self.0.iter().map(|h| h.cover).sum::<f32>() / self.0.len().max(1) as f32
    }
}

/// Sample the 64x64 landuse grid (`LU_RESIDENTIAL`, `LU_COMMERCIAL`, `LU_INDUSTRIAL`, `LU_OPEN`)
/// at normalized tile coordinates `(u, v)` in `0..1`.
pub fn sample_landuse(landuse: &[u8], u: f32, v: f32) -> u8 {
    if landuse.len() < 64 * 64 {
        return 0;
    }
    let x = ((u * 64.0).floor() as isize).clamp(0, 63) as usize;
    let y = ((v * 64.0).floor() as isize).clamp(0, 63) as usize;
    landuse[y * 64 + x]
}

/// Test a circle of `radius` metres at normalized tile coordinates `p_uv` (`u` east, `v` south)
/// against a closed footprint `ring` (also in `0..1` normalized tile coordinates), where the tile
/// spans `tile_m` metres across.
///
/// Returns `Some((outward_normal_en, penetration_depth_m))` in the local East-North frame (`+x` East, `+y` North).
pub fn collide_ring_circle(p_uv: [f32; 2], radius: f32, tile_m: f32, ring: &[[f32; 2]]) -> Option<(Vec2, f32)> {
    let n = ring.len();
    if n < 3 || tile_m <= 1e-3 {
        return None;
    }
    let inside = point_in_ring(p_uv, ring);
    let p = Vec2::new(p_uv[0], p_uv[1]);

    let mut signed_area2 = 0.0f32;
    for i in 0..n {
        let a = Vec2::from(ring[i]);
        let b = Vec2::from(ring[(i + 1) % n]);
        signed_area2 += a.perp_dot(b);
    }
    let wind_sign = if signed_area2 >= 0.0 { 1.0f32 } else { -1.0f32 };

    let mut best_d2 = f32::MAX;
    let mut best_delta_en = Vec2::ZERO;
    let mut best_edge_out_en = Vec2::X;

    for i in 0..n {
        let a = Vec2::from(ring[i]);
        let b = Vec2::from(ring[(i + 1) % n]);
        let ab = b - a;
        let len2 = ab.length_squared();
        if len2 < 1e-12 {
            continue;
        }
        let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
        let q = a + ab * t;
        let d_uv = p - q;
        let delta_en = Vec2::new(d_uv.x * tile_m, -d_uv.y * tile_m);
        let d2 = delta_en.length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_delta_en = delta_en;
            let edge_en = Vec2::new(ab.x, -ab.y).normalize_or_zero();
            best_edge_out_en = Vec2::new(-edge_en.y, edge_en.x) * wind_sign;
        }
    }

    let dist = best_d2.sqrt();
    if inside {
        let normal_en = if dist > 1e-4 { -best_delta_en / dist } else { best_edge_out_en };
        Some((normal_en, radius + dist))
    } else if dist < radius {
        let normal_en = if dist > 1e-4 { best_delta_en / dist } else { best_edge_out_en };
        Some((normal_en, radius - dist))
    } else {
        None
    }
}

/// Find the closest drivable lane point and forward East-North tangent in `net`
/// to planet-frame position `pos`, given the tile `center`.
/// Returns `Some((dist_sq_m2, lane_pos_ecef, forward_en))`.
pub fn nearest_lane_in_net(center: DVec3, net: &crate::roads::RoadNet, pos: DVec3) -> Option<(f32, DVec3, Vec2)> {
    if net.paths.is_empty() {
        return None;
    }
    let lp = (pos - center).as_vec3();
    let up = center.normalize().as_vec3();
    let (east, north, _) = geo::enu(pos);

    let mut best_d2 = f32::MAX;
    let mut best_pose: Option<(DVec3, Vec2)> = None;

    for p in &net.paths {
        if p.hidden || p.lanes == 0 || p.pts.len() < 2 || p.len < 1.0 {
            continue;
        }
        for i in 0..p.pts.len() - 1 {
            let a = p.pts[i];
            let b = p.pts[i + 1];
            let ab = b - a;
            let seg_len2 = ab.length_squared();
            if seg_len2 < 1e-6 {
                continue;
            }
            let t = ((lp - a).dot(ab) / seg_len2).clamp(0.0, 1.0);
            let s = (p.cum[i] + (p.cum[i + 1] - p.cum[i]) * t).clamp(0.0, p.len);
            let (center_pt, tan) = p.sample(s);
            let dir = tan.normalize_or_zero();
            if dir == Vec3::ZERO {
                continue;
            }
            let right = dir.cross(up).normalize_or_zero();
            let lane_idx = 0u8;
            let candidates: &[f32] = if p.oneway { &[1.0] } else { &[1.0, -1.0] };
            for &dir_sign in candidates {
                let travel_dir = dir * dir_sign;
                let travel_right = right * dir_sign;
                let off = if p.oneway {
                    (lane_idx as f32 + 0.5 - p.lanes as f32 * 0.5) * p.lane_w
                } else {
                    (lane_idx as f32 + 0.5) * p.lane_w * if net.left { -1.0 } else { 1.0 }
                };
                let lane_local = center_pt + travel_right * off;
                let d2 = (lane_local - lp).length_squared();
                if d2 < best_d2 {
                    best_d2 = d2;
                    let world_pos = center + lane_local.as_dvec3();
                    let td = travel_dir.as_dvec3();
                    let fwd_en = Vec2::new(td.dot(east) as f32, td.dot(north) as f32).normalize_or_zero();
                    let fwd_en = if fwd_en == Vec2::ZERO { Vec2::Y } else { fwd_en };
                    best_pose = Some((world_pos, fwd_en));
                }
            }
        }
    }

    best_pose.map(|(wpos, fwd)| (best_d2, wpos, fwd))
}

impl BuildingData {
    /// Bilinearly sample neighbourhood metrics (`cover`, `green`, `wealth`, `mean_h`) at `(lat, lon)`.
    pub fn hood_at(&self, lat: f64, lon: f64) -> Hood {
        let (u, v) = self.id.frac(lat, lon);
        self.hoods.sample_uv(u as f32, v as f32)
    }

    /// Sample the landuse classification (`LU_*`) at `(lat, lon)`.
    pub fn landuse_at(&self, lat: f64, lon: f64) -> u8 {
        let (u, v) = self.id.frac(lat, lon);
        sample_landuse(&self.landuse, u as f32, v as f32)
    }

    /// Snap a planet-frame position `pos` to the nearest road lane in this tile,
    /// returning `(lane_pos_ecef, forward_en)`.
    pub fn nearest_road_spawn(&self, pos: DVec3) -> Option<(DVec3, Vec2)> {
        nearest_lane_in_net(self.center, &self.net, pos).map(|(_, p, fwd)| (p, fwd))
    }
}

fn neighbourhoods(cx: &Ctx, vt: &VecTile) -> Hoods {
    let mut area = vec![0f32; NB * NB];
    let mut house = vec![(0f32, 0f32); NB * NB]; // count, total area of house-sized footprints
    let mut hsum = vec![(0f32, 0f32); NB * NB];
    for b in &vt.buildings {
        let r = &b.rings[0];
        let n = r.len() as f32;
        let c = r.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n]);
        if c[0] < 0.0 || c[1] < 0.0 || c[0] >= cx.ext || c[1] >= cx.ext {
            continue;
        }
        let a = crate::mvt::signed_area(r).abs() * cx.unit_m * cx.unit_m;
        let i = ((c[1] / cx.ext * NB as f32) as usize).min(NB - 1) * NB + ((c[0] / cx.ext * NB as f32) as usize).min(NB - 1);
        area[i] += a;
        hsum[i].0 += b.height * a;
        hsum[i].1 += a;
        if a > 35.0 && a < 700.0 && b.height < 14.0 {
            house[i].0 += 1.0;
            house[i].1 += a;
        }
    }
    let cell_m2 = (cx.ext * cx.unit_m / NB as f32).powi(2);
    let mut green = vec![0f32; NB * NB];
    let step = ATLAS / NB;
    for j in 0..NB {
        for i in 0..NB {
            let mut g = 0.0;
            for y in (0..step).step_by(2) {
                for x in (0..step).step_by(2) {
                    let px = &cx.atlas[((j * step + y) * ATLAS + i * step + x) * 4..];
                    g += greenness([px[0], px[1], px[2]]);
                }
            }
            green[j * NB + i] = g / (step * step / 4) as f32;
        }
    }
    let mean_elev = cx.ground([cx.ext * 0.5, cx.ext * 0.5]);
    let mut raw = vec![Hood::default(); NB * NB];
    for j in 0..NB {
        for i in 0..NB {
            let k = j * NB + i;
            let cover = (area[k] / cell_m2).min(1.0);
            let size = if house[k].0 > 0.0 { house[k].1 / house[k].0 } else { 110.0 };
            // Bigger homes, more green and a bit of a view read as money;
            // tiny homes packed wall to wall read as the opposite.
            let size_t = ((size - 55.0) / 230.0).clamp(0.0, 1.0);
            let green_t = ((green[k] - 0.04) / 0.3).clamp(0.0, 1.0);
            let c = [(i as f32 + 0.5) / NB as f32 * cx.ext, (j as f32 + 0.5) / NB as f32 * cx.ext];
            let hill = ((cx.ground(c) - mean_elev) / 120.0).clamp(-0.5, 1.0);
            let crowd = if size < 80.0 { ((cover - 0.3) / 0.3).clamp(0.0, 1.0) } else { 0.0 };
            let wealth = (0.22 + 0.5 * size_t + 0.28 * green_t + 0.12 * hill - 0.3 * crowd).clamp(0.0, 1.0);
            raw[k] = Hood { cover, green: green[k], wealth, mean_h: if hsum[k].1 > 0.0 { hsum[k].0 / hsum[k].1 } else { 0.0 } };
        }
    }
    // Blur wealth so it changes over neighbourhoods rather than cell by cell.
    let mut out = raw.clone();
    for j in 0..NB {
        for i in 0..NB {
            let (mut s, mut n) = (0.0, 0.0);
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let (x, y) = (i as i32 + di, j as i32 + dj);
                    if x >= 0 && y >= 0 && x < NB as i32 && y < NB as i32 {
                        let w = if di == 0 && dj == 0 { 2.0 } else { 1.0 };
                        s += raw[y as usize * NB + x as usize].wealth * w;
                        n += w;
                    }
                }
            }
            out[j * NB + i].wealth = s / n;
        }
    }
    Hoods(out)
}

// ---------------------------------------------------------------- styling

fn pick(list: &[[u8; 3]], h: f32) -> [u8; 3] {
    list[((h * list.len() as f32) as usize).min(list.len() - 1)]
}
fn mixc(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [0, 1, 2].map(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).clamp(0.0, 255.0) as u8)
}

const PAL_NA: &[[u8; 3]] = &[[226, 222, 212], [205, 196, 178], [188, 190, 186], [214, 205, 172], [160, 175, 186], [172, 182, 160], [222, 208, 160], [150, 92, 74], [196, 170, 146], [236, 232, 226], [140, 150, 160], [206, 178, 170]];
const PAL_MED: &[[u8; 3]] = &[[236, 230, 214], [226, 206, 168], [214, 170, 132], [222, 190, 160], [240, 236, 228], [204, 180, 140], [228, 214, 190]];
const PAL_NEUR: &[[u8; 3]] = &[[150, 86, 66], [168, 104, 78], [196, 170, 120], [226, 222, 210], [170, 168, 160], [122, 80, 62], [206, 196, 176], [188, 150, 110]];
const PAL_ARID: &[[u8; 3]] = &[[214, 196, 164], [228, 218, 198], [190, 184, 172], [222, 196, 176], [200, 180, 150], [236, 230, 218]];
const PAL_TROPIC: &[[u8; 3]] = &[[186, 184, 176], [200, 214, 220], [208, 222, 196], [226, 200, 190], [226, 216, 180], [170, 150, 126], [232, 228, 216]];
const PAL_EASIA: &[[u8; 3]] = &[[206, 204, 196], [226, 222, 212], [190, 176, 156], [160, 160, 158], [214, 200, 178], [176, 150, 130]];
const PAL_OFFICE: &[[u8; 3]] = &[[196, 192, 184], [170, 168, 164], [210, 200, 180], [128, 126, 124], [150, 110, 90], [222, 220, 214], [176, 160, 140]];
const PAL_GLASS: &[[u8; 3]] = &[[110, 140, 165], [95, 125, 135], [120, 150, 150], [70, 85, 100], [140, 120, 95], [130, 160, 185], [60, 70, 80]];
const PAL_SHED: &[[u8; 3]] = &[[190, 192, 190], [168, 180, 188], [206, 200, 180], [150, 164, 150], [212, 212, 208], [140, 144, 150], [176, 150, 130]];

fn wall_palette(r: Region) -> &'static [[u8; 3]] {
    match r {
        Region::NAm | Region::Oceania => PAL_NA,
        Region::Med | Region::LatAm => PAL_MED,
        Region::NEur => PAL_NEUR,
        Region::Mena | Region::SAsia => PAL_ARID,
        Region::Africa | Region::SeAsia => PAL_TROPIC,
        Region::EAsia => PAL_EASIA,
    }
}

/// Chance that a house here has a pitched roof at all.
fn pitched_chance(r: Region) -> f32 {
    match r {
        Region::NAm => 0.92,
        Region::LatAm => 0.4,
        Region::NEur => 0.97,
        Region::Med => 0.8,
        Region::Mena => 0.08,
        Region::Africa => 0.55,
        Region::SAsia => 0.22,
        Region::EAsia => 0.7,
        Region::SeAsia => 0.7,
        Region::Oceania => 0.95,
    }
}

/// Roof material (shader id) and its typical colour.
fn roof_stuff(r: Region, wealth: f32, h: f32) -> (u8, [u8; 3]) {
    let tin = (4u8, pick(&[[150, 150, 148], [120, 124, 128], [140, 96, 70], [110, 60, 44], [170, 176, 180]], h));
    let shingle = (0u8, pick(&[[84, 80, 78], [104, 96, 88], [70, 70, 74], [120, 100, 84], [92, 72, 60], [130, 126, 120]], h));
    let clay = (1u8, pick(&[[176, 96, 62], [160, 84, 56], [190, 112, 74], [148, 80, 60]], h));
    let dark_tile = (1u8, pick(&[[110, 62, 50], [84, 70, 66], [128, 74, 54], [70, 66, 68]], h));
    let slate = (3u8, pick(&[[70, 74, 82], [86, 88, 92], [60, 62, 68]], h));
    match r {
        Region::NAm | Region::Oceania => if wealth < 0.18 && h < 0.4 { tin } else if wealth > 0.75 && h < 0.25 { clay } else { shingle },
        Region::Med | Region::LatAm => if wealth < 0.3 { tin } else { clay },
        Region::NEur => if h < 0.3 { slate } else { dark_tile },
        Region::Africa | Region::SeAsia | Region::SAsia | Region::Mena => if wealth > 0.6 { clay } else { tin },
        Region::EAsia => if wealth < 0.25 { tin } else if h < 0.5 { dark_tile } else { slate },
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum Kind {
    Shed,
    House,
    Row,
    MidRise,
    Tower,
    Industrial,
    Retail,
    Part,
}

struct Style {
    kind: Kind,
    roof: Roof,
    roof_mat: u8,
    roof_col: [u8; 3],
    /// covering of a flat roof (F_*)
    flat_mat: u8,
    wall: [u8; 3],
    facade: u8,
    wealth: f32,
    parapet: f32,
}

/// The real roof colour, read from the satellite image over the inner part of
/// the footprint. The image never lines up exactly with the map and trees
/// overhang roofs, so samples that look like canopy or cast shadow are
/// dropped, and the rest averaged with the brightest and darkest quarters
/// trimmed off (vents, skylights, the odd misplaced bit of street).
fn roof_colour(cx: &Ctx, ring: &[[f32; 2]], cen: [f32; 2]) -> [u8; 3] {
    let lum = |c: &[u8; 3]| (c[0] as f32 + c[1] as f32 + c[2] as f32) / 3.0;
    let mut all: Vec<[u8; 3]> = Vec::with_capacity(ring.len() * 3 + 3);
    for p in ring.iter().chain(std::iter::once(&cen)) {
        for t in [0.15f32, 0.4, 0.65] {
            all.push(cx.img([cen[0] + (p[0] - cen[0]) * t, cen[1] + (p[1] - cen[1]) * t]));
        }
    }
    let mean = all.iter().map(lum).sum::<f32>() / all.len() as f32;
    let floor = (mean * 0.6).min(70.0);
    let mut keep: Vec<[u8; 3]> = all.iter().copied().filter(|c| greenness(*c) < 0.3 && lum(c) > floor).collect();
    let dirty = keep.len() < 3;
    if dirty {
        keep = all;
    }
    keep.sort_by(|a, b| lum(a).total_cmp(&lum(b)));
    let n = keep.len();
    let mid = &keep[n / 4..(n - n / 4).max(n / 4 + 1)];
    let c = [0, 1, 2].map(|k| mid.iter().map(|c| c[k] as f32).sum::<f32>() / mid.len() as f32);
    if dirty {
        // Mostly canopy or shadow: keep only how light it is.
        let l = (c[0] + c[1] + c[2]) / 3.0;
        return [l as u8; 3];
    }
    c.map(|x| x as u8)
}

/// Flat roof coverings (K_FLAT sty[1] low bits); must match the shader.
const F_MEMBRANE: u8 = 0;
const F_BITUMEN: u8 = 1;
const F_GRAVEL: u8 = 2;
const F_PAVERS: u8 = 3;

/// What covers a flat roof, chosen to agree with how light it looks from above.
fn flat_stuff(kind: Kind, lum: f32, r: f32) -> (u8, [u8; 3]) {
    let q = (r * 7.0).fract();
    let membrane = (F_MEMBRANE, pick(&[[204, 204, 198], [190, 192, 190], [176, 176, 170]], q));
    let bitumen = (F_BITUMEN, pick(&[[64, 63, 62], [86, 82, 78], [112, 108, 100]], q));
    let gravel = (F_GRAVEL, pick(&[[150, 144, 134], [132, 128, 122], [160, 152, 140]], q));
    let pavers = (F_PAVERS, pick(&[[176, 170, 160], [156, 152, 146], [186, 168, 150]], q));
    match kind {
        Kind::Tower if r < 0.45 => pavers,
        Kind::Industrial | Kind::Retail if lum > 110.0 => membrane,
        _ if lum > 150.0 => membrane,
        _ if lum < 85.0 => bitumen,
        Kind::Tower | Kind::MidRise if r < 0.5 => gravel,
        _ if r < 0.35 => gravel,
        _ if lum > 120.0 => membrane,
        _ => bitumen,
    }
}

/// The rule set: from what we know about a footprint to what gets built.
#[allow(clippy::too_many_arguments)]
fn style(cx: &Ctx, b: &crate::sources::BuildingPoly, area: f32, hl: f32, hw: f32, rect: f32, nverts: usize, hood: Hood, landuse: u8, img: [u8; 3], seed: u32) -> Style {
    let r = |k: u32| hash(seed, k);
    let h = b.height;
    let reg = cx.region;
    // Each building strays from its neighbourhood: some are kept perfectly, some are not.
    let wealth = (hood.wealth + (r(1) - 0.5) * 0.5 + if r(2) < 0.08 { -0.35 } else { 0.0 }).clamp(0.0, 1.0);
    let urban = hood.cover > 0.3;
    let kind = if b.min_height > 0.5 {
        Kind::Part
    } else if area < 32.0 && h < 5.5 {
        Kind::Shed
    } else if h >= 45.0 || (h >= 28.0 && area > 500.0) {
        Kind::Tower
    } else if landuse == LU_INDUSTRIAL && h < 22.0 || (area > 1500.0 && h < 13.0 && landuse != LU_RESIDENTIAL && hl / hw < 6.0 && rect > 0.6) {
        if landuse == LU_COMMERCIAL { Kind::Retail } else { Kind::Industrial }
    } else if h >= 14.0 {
        Kind::MidRise
    } else if area < 620.0 {
        if urban && (hw < 5.5 || hood.cover > 0.42) { Kind::Row } else { Kind::House }
    } else if landuse == LU_COMMERCIAL || urban {
        Kind::Retail
    } else {
        Kind::House
    };
    let lum = (img[0] as f32 + img[1] as f32 + img[2] as f32) / 3.0;
    let palette = wall_palette(reg);
    let mut wall = pick(palette, r(3));
    // Poorer streets fade toward bare render and concrete; rich ones stay crisp.
    wall = mixc(wall, [150, 146, 138], (0.45 - wealth).max(0.0) * 0.9);
    wall = mixc(wall, [240, 238, 232], (wealth - 0.7).max(0.0) * 0.5);
    let (mut roof_mat, mut roof_col) = roof_stuff(reg, wealth, r(4));
    let complex = nverts > 8 || rect < 0.72;
    let pitch = match reg {
        Region::NEur => 0.75 + r(5) * 0.5,
        Region::Med | Region::LatAm | Region::Mena => 0.32 + r(5) * 0.16,
        _ => 0.42 + r(5) * 0.35,
    };
    let mut parapet = 0.0;
    let mut facade = 0u8;
    let roof = match kind {
        Kind::Part => Roof::Flat,
        Kind::Shed => {
            roof_mat = 4;
            if r(6) < 0.5 { Roof::Shed(0.18) } else { Roof::Gable(hw * 0.35, 0.0) }
        }
        Kind::House => {
            let p = pitched_chance(reg) * if complex && area > 300.0 { 0.6 } else { 1.0 };
            if r(6) > p {
                parapet = 0.35 + r(7) * 0.4;
                Roof::Flat
            } else if complex {
                // Awkward footprints get a hipped skirt with a flat middle.
                Roof::Mansard(pitch.min(0.7), (hw * 0.45 * pitch).min(2.6))
            } else {
                let t = r(8);
                // Money buys hips and steeper, tidier roofs; the far end makes do with a lean-to.
                if wealth < 0.22 && t < 0.45 {
                    Roof::Shed(0.14 + r(9) * 0.1)
                } else if t < 0.3 + wealth * 0.35 && hl / hw < 2.6 {
                    Roof::Hip(pitch)
                } else if t > 0.9 {
                    Roof::Gable(hw * pitch * 0.9, hw * (r(9) - 0.5) * 0.7)
                } else {
                    Roof::Gable(hw * pitch, 0.0)
                }
            }
        }
        Kind::Row => {
            facade = 1;
            let p = match reg {
                Region::NEur => 0.9,
                Region::NAm => 0.3,
                Region::EAsia | Region::Oceania => 0.5,
                _ => 0.25,
            };
            if r(6) < p && !complex {
                Roof::Gable(hw * pitch * 0.8, 0.0)
            } else {
                parapet = 0.5 + r(7) * 0.7;
                Roof::Flat
            }
        }
        Kind::MidRise => {
            facade = if landuse == LU_COMMERCIAL { if r(6) < 0.5 { 2 } else { 5 } } else if matches!(reg, Region::NEur | Region::NAm) && r(6) < 0.45 { 6 } else if r(6) < 0.8 { 1 } else { 5 };
            parapet = 0.6 + r(7) * 0.6;
            if reg == Region::NEur && r(8) < 0.35 && !complex { Roof::Mansard(1.6, 2.6) } else { Roof::Flat }
        }
        Kind::Tower => {
            let glassy = (0.25 + (h - 45.0) / 160.0 + if landuse == LU_COMMERCIAL { 0.2 } else { 0.0 }).min(0.85);
            facade = if r(6) < glassy { 3 } else if landuse == LU_RESIDENTIAL || r(7) < 0.4 { 6 } else { 2 };
            parapet = 0.9;
            Roof::Flat
        }
        Kind::Industrial => {
            facade = 4;
            roof_mat = 2;
            roof_col = pick(&[[170, 172, 174], [150, 156, 160], [196, 196, 190], [130, 140, 150], [150, 120, 100]], r(4));
            wall = pick(PAL_SHED, r(3));
            if complex {
                parapet = 0.4;
                Roof::Flat
            } else if hw > 14.0 && r(6) < 0.55 {
                Roof::Multi(((hw / 9.0) as u32).clamp(2, 6), 1.2 + r(7) * 1.2)
            } else if r(6) < 0.85 {
                Roof::Gable((hw * 0.16).min(4.0), 0.0)
            } else {
                parapet = 0.4;
                Roof::Flat
            }
        }
        Kind::Retail => {
            facade = if h > 7.0 || r(6) < 0.6 { 5 } else { 2 };
            parapet = 0.7 + r(7) * 0.7;
            if !complex && area < 1200.0 && r(8) < 0.12 { Roof::Gable(hw * 0.3, 0.0) } else { Roof::Flat }
        }
    };
    if matches!(kind, Kind::MidRise | Kind::Tower | Kind::Retail | Kind::Part) {
        wall = pick(if facade == 3 { PAL_GLASS } else if facade == 6 { PAL_NEUR } else { PAL_OFFICE }, r(3));
    }
    if kind == Kind::Row && reg == Region::NAm {
        // Painted town houses.
        wall = pick(PAL_NA, r(3));
    }
    // A tagged colour always wins; otherwise nudge toward what the roof looks like from above.
    if let Some(c) = b.colour {
        wall = c;
    } else if facade != 3 {
        wall = mixc(wall, [lum as u8; 3], 0.12);
    }
    // Flat roofs take their covering from how light they look, and most of their colour from the image.
    let (flat_mat, flat_col) = flat_stuff(kind, lum, r(10));
    let roof_col = if matches!(roof, Roof::Flat) { mixc(flat_col, img, 0.55) } else { mixc(roof_col, img, 0.4) };
    Style { kind, roof, roof_mat, roof_col, flat_mat, wall, facade, wealth, parapet }
}

// --------------------------------------------------------------- generator

struct Shape {
    /// rings in metres (outer first)
    rings: Vec<Vec<Vec2>>,
    c: Vec2,
    u: Vec2,
    v: Vec2,
    hl: f32,
    hw: f32,
    gmin: f32,
}
impl Shape {
    fn uv(&self, p: Vec2) -> (f32, f32) {
        ((p - self.c).dot(self.u), (p - self.c).dot(self.v))
    }
}

/// Walls of one ring from `z0` up to `top` + roof height, following the roof's folds.
#[allow(clippy::too_many_arguments)]
fn walls(cx: &Ctx, m: &mut Mesh, lvl: Lvl, sh: &Shape, ring: &[Vec2], z0: f32, top: f32, roof: &Roof, creases: &[(f32, f32, f32)], col: [u8; 4], sty: [u8; 4], extra: f32) {
    let mut along = 0.0;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let l = (b - a).length();
        if l < 0.05 {
            continue;
        }
        let (ua, ub) = (sh.uv(a), sh.uv(b));
        let mut ts = vec![0.0f32, 1.0];
        for c in creases {
            let (sa, sb) = (c.0 * ua.0 + c.1 * ua.1 - c.2, c.0 * ub.0 + c.1 * ub.1 - c.2);
            if sa * sb < 0.0 {
                ts.push(sa / (sa - sb));
            }
        }
        ts.sort_by(|x, y| x.total_cmp(y));
        for w in ts.windows(2) {
            if w[1] - w[0] < 1e-4 {
                continue;
            }
            let (p0, p1) = (a + (b - a) * w[0], a + (b - a) * w[1]);
            let h = |p: Vec2| {
                let (u, v) = sh.uv(p);
                top + roof.h(u, v, sh.hl, sh.hw) + extra
            };
            let (h0, h1) = (h(p0), h(p1));
            let (s0, s1) = (along + l * w[0], along + l * w[1]);
            m.poly(lvl, &[(cx.pos_m(p0, z0), [s0, z0 - sh.gmin]), (cx.pos_m(p1, z0), [s1, z0 - sh.gmin]), (cx.pos_m(p1, h1), [s1, h1 - sh.gmin]), (cx.pos_m(p0, h0), [s0, h0 - sh.gmin])], col, sty, None);
        }
        along += l;
    }
}

/// Roof surface over a set of rings (outer + holes).
fn roof_surface(cx: &Ctx, m: &mut Mesh, lvl: Lvl, sh: &Shape, rings: &[Vec<Vec2>], top: f32, roof: &Roof, creases: &[(f32, f32, f32)], col: [u8; 4], sty: [u8; 4]) {
    let mut flat: Vec<f64> = Vec::new();
    let mut holes = Vec::new();
    let mut pts = Vec::new();
    for (k, r) in rings.iter().enumerate() {
        if k > 0 {
            holes.push(flat.len() / 2);
        }
        for p in r {
            let (u, v) = sh.uv(*p);
            flat.extend([u as f64, v as f64]);
            pts.push(Vec2::new(u, v));
        }
    }
    let Ok(ix) = earcutr::earcut(&flat, &holes, 2) else { return };
    let mut tris: Vec<[Vec2; 3]> = ix.chunks_exact(3).map(|t| [pts[t[0]], pts[t[1]], pts[t[2]]]).collect();
    for c in creases {
        tris = split_tris(tris, *c);
    }
    let upv = (cx.pos_m(sh.c, top) + cx.center.as_vec3()).normalize();
    for t in tris {
        let hs = t.map(|p| roof.h(p.x, p.y, sh.hl, sh.hw));
        // Slope direction of this facet, for laying tiles in rows.
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let det = e1.perp_dot(e2);
        let grad = if det.abs() > 1e-6 { Vec2::new((hs[1] - hs[0]) * e2.y - (hs[2] - hs[0]) * e1.y, (hs[2] - hs[0]) * e1.x - (hs[1] - hs[0]) * e2.x) / det } else { Vec2::ZERO };
        let g = grad.length();
        let p3: Vec<(Vec3, [f32; 2])> = (0..3)
            .map(|k| {
                let world = sh.c + sh.u * t[k].x + sh.v * t[k].y;
                let uv = if g > 0.02 { [t[k].dot(grad.perp() / g), -hs[k] / g * (1.0 + g * g).sqrt()] } else { [t[k].x, t[k].y] };
                (cx.pos_m(world, top + hs[k]), uv)
            })
            .collect();
        m.poly(lvl, &p3, col, sty, Some(upv));
    }
}

fn kcol(c: [u8; 3], k: u8) -> [u8; 4] {
    [c[0], c[1], c[2], k]
}

/// Material of a K_PLAIN surface (sty[1]); must match the shader.
const P_PLAIN: u8 = 0;
const P_HVAC: u8 = 1;
const P_TANK: u8 = 2;
const P_SOLAR: u8 = 3;
const P_SKYLIGHT: u8 = 4;
const P_STAIR: u8 = 5;
const P_AWNING: u8 = 6;
const P_AC: u8 = 7;
/// K_LAMP sty[1]: aircraft warning light.
const L_WARN: u8 = 2;
/// K_FLAT sty[1] flag: a helipad is painted on this roof (centred on the shape).
pub const HELIPAD: u8 = 16;

/// Equipment on a flat roof: the stair and lift head, air handlers, and close
/// up the smaller things - timber water tanks, rows of solar panels,
/// skylights, aircraft warning lights.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn roof_clutter(cx: &Ctx, m: &mut Mesh, sh: &Shape, ring: &[Vec2], top: f32, st: &Style, seed: u32, height: f32, helipad: bool, avoid: &[Vec<Vec2>]) {
    let area = sh.hl * sh.hw * 4.0;
    let want = match st.kind {
        Kind::Tower => 3 + (area / 500.0) as usize,
        Kind::MidRise | Kind::Retail => 1 + (area / 260.0) as usize,
        Kind::Industrial => (area / 700.0) as usize,
        Kind::Row => (hash(seed, 60) < 0.5) as usize,
        _ => 0,
    }
    .min(14);
    let mut placed: Vec<(Vec2, f32)> = Vec::new();
    if helipad {
        placed.push((sh.c, 10.0));
    }
    let s8 = (hash(seed, 61) * 255.0) as u8;
    let w8 = (st.wealth * 255.0) as u8;
    let sty = |mat: u8| [s8, mat, w8, 0];
    let fits = |c: Vec2, hx: f32, hy: f32, placed: &Vec<(Vec2, f32)>| {
        let corners = [c + sh.u * hx + sh.v * hy, c - sh.u * hx + sh.v * hy, c + sh.u * hx - sh.v * hy, c - sh.u * hx - sh.v * hy];
        // Keep clear of holes and of any part standing on or rising through the roof.
        let (ex, ey) = (hx + 0.4, hy + 0.4);
        let clear = [c, c + sh.u * ex + sh.v * ey, c - sh.u * ex + sh.v * ey, c + sh.u * ex - sh.v * ey, c - sh.u * ex - sh.v * ey];
        corners.iter().all(|p| in_ring_m(*p, ring)) && !sh.rings.iter().skip(1).chain(avoid).any(|h| clear.iter().any(|p| in_ring_m(*p, h))) && !placed.iter().any(|(p, pr)| (*p - c).length() < hx.hypot(hy) + pr + 0.6)
    };
    let mut n_box = 0;
    for k in 0..want * 4 {
        if n_box >= want {
            break;
        }
        let r = |j: u32| hash(seed, 100 + k as u32 * 8 + j);
        let (u, v) = ((r(0) * 2.0 - 1.0) * (sh.hl - 1.5), (r(1) * 2.0 - 1.0) * (sh.hw - 1.5));
        let c = sh.c + sh.u * u + sh.v * v;
        // First box on a tall building is the lift overrun / stair head, the rest are plant.
        let big = n_box == 0 && matches!(st.kind, Kind::Tower | Kind::MidRise);
        let (hx, hy, hz) = if big { (2.0 + r(2) * 2.5, 1.6 + r(3) * 1.6, 2.6 + r(4) * 1.5) } else { (0.6 + r(2) * 1.3, 0.5 + r(3) * 0.9, 0.7 + r(4) * 1.1) };
        let (hx, hy) = (hx.min(sh.hl * 0.4), hy.min(sh.hw * 0.4));
        if !fits(c, hx, hy, &placed) {
            continue;
        }
        placed.push((c, hx.hypot(hy)));
        n_box += 1;
        if big {
            m.boxm(cx, false, c, sh.u * hx, sh.v * hy, top - 0.1, top + hz, kcol(mixc(st.wall, [150, 150, 150], 0.5), K_PLAIN), sty(P_STAIR), None);
            if st.kind == Kind::Tower && height > 110.0 && r(6) < 0.4 {
                // Mast.
                let mh = height * (0.12 + r(7) * 0.15);
                m.boxm(cx, false, c, sh.u * 0.35, sh.v * 0.35, top + hz, top + hz + mh, kcol([120, 122, 126], K_PLAIN), sty(P_PLAIN), None);
                m.boxm(cx, true, c, sh.u * 0.3, sh.v * 0.3, top + hz + mh, top + hz + mh + 0.4, [255, 40, 30, K_LAMP], [s8, L_WARN, 0, 0], None);
            }
        } else {
            let col = pick(&[[168, 170, 172], [140, 144, 146], [196, 196, 192], [120, 124, 122]], r(5));
            m.boxm(cx, true, c, sh.u * hx, sh.v * hy, top - 0.1, top + hz, kcol(col, K_PLAIN), sty(P_HVAC), None);
        }
    }
    // Warning lights on the corners of anything tall.
    if height > 70.0 {
        if let Some(inner) = inset(ring, 0.5) {
            let step = (inner.len() / 4).max(1);
            for p in inner.iter().step_by(step).take(4) {
                m.boxm(cx, true, *p, sh.u * 0.2, sh.v * 0.2, top + 0.8, top + 1.3, [255, 40, 30, K_LAMP], [s8.wrapping_add(p.x as u8), L_WARN, 0, 0], None);
            }
        }
    }
    if !m.fine_ok() {
        return;
    }
    // Timber water tanks on legs: old American mid-rises.
    if cx.region == Region::NAm && matches!(st.kind, Kind::MidRise | Kind::Tower) && height < 130.0 && hash(seed, 70) < 0.45 + (0.5 - st.wealth) * 0.4 {
        for k in 0..1 + (hash(seed, 71) < 0.3) as u32 {
            let r = |j: u32| hash(seed, 72 + k * 8 + j);
            let rad = 1.5 + r(0) * 1.0;
            let c = sh.c + sh.u * (r(1) * 2.0 - 1.0) * (sh.hl - rad - 1.0) + sh.v * (r(2) * 2.0 - 1.0) * (sh.hw - rad - 1.0);
            if !fits(c, rad, rad, &placed) {
                continue;
            }
            placed.push((c, rad * 1.4));
            let legs = 2.4 + r(3) * 1.5;
            for (a, b) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                let lp = c + (sh.u * a + sh.v * b) * (rad * 0.62);
                m.boxm(cx, FINE, lp, sh.u * 0.12, sh.v * 0.12, top - 0.1, top + legs, kcol([60, 58, 56], K_PLAIN), sty(P_PLAIN), None);
            }
            let th = rad * 1.6 + r(4) * 1.5;
            m.prism(cx, FINE, c, rad, 10, top + legs, top + legs + th, kcol([110, 80, 58], K_PLAIN), sty(P_TANK), None);
            // Conical lid.
            m.prism(cx, FINE, c, rad * 0.55, 10, top + legs + th, top + legs + th + rad * 0.45, kcol([70, 62, 56], K_PLAIN), sty(P_TANK), None);
        }
    }
    // Solar panels in tilted rows, facing the equator.
    let solar_p = match st.kind {
        Kind::Retail | Kind::Industrial => 0.25 + 0.35 * st.wealth,
        Kind::MidRise => 0.1 + 0.3 * st.wealth,
        _ => 0.0,
    } + if matches!(cx.region, Region::Oceania | Region::NEur | Region::Med) { 0.15 } else { 0.0 };
    if hash(seed, 80) < solar_p && sh.hl > 5.0 && sh.hw > 4.0 {
        // "South" in the metric frame is +y in the north, -y in the south.
        let eq = Vec2::new(0.0, if cx.lat >= 0.0 { 1.0 } else { -1.0 });
        let (along, across) = if sh.v.dot(eq).abs() > sh.u.dot(eq).abs() { (sh.u, sh.v * sh.v.dot(eq).signum()) } else { (sh.v, sh.u * sh.u.dot(eq).signum()) };
        let (la, lc) = if along == sh.u { (sh.hl, sh.hw) } else { (sh.hw, sh.hl) };
        let rows = (((lc - 2.0) * 2.0) / 2.6) as i32;
        let half = (la - 1.8).min(14.0);
        let tilt = 0.9f32;
        let rise = tilt * 0.45;
        for i in 0..rows.min(16) {
            let off = -lc + 1.6 + i as f32 * 2.6;
            let c = sh.c + across * (off + 0.6);
            if !fits(c, half.min(la - 1.0), 0.8, &placed) || !m.fine_ok() {
                continue;
            }
            let (f0, f1) = (c - across * 0.45, c + across * 0.45);
            let z0 = top + 0.35;
            let quad = [(f0 - along * half, z0 + rise), (f0 + along * half, z0 + rise), (f1 + along * half, z0), (f1 - along * half, z0)];
            let pts: Vec<(Vec3, [f32; 2])> = quad.iter().enumerate().map(|(k, (p, z))| (cx.pos_m(*p, *z), [if k == 1 || k == 2 { half * 2.0 } else { 0.0 }, if k >= 2 { tilt } else { 0.0 }])).collect();
            let upv = (pts[0].0 + cx.center.as_vec3()).normalize();
            m.poly(FINE, &pts, kcol([20, 26, 50], K_PLAIN), sty(P_SOLAR), Some(upv));
        }
    }
    // Skylights over sheds and shops.
    if matches!(st.kind, Kind::Industrial | Kind::Retail) && hash(seed, 85) < 0.5 {
        let nx = ((sh.hl - 3.0) / 6.0) as i32;
        let ny = ((sh.hw - 3.0) / 6.0) as i32;
        for i in -nx..=nx {
            for j in -ny..=ny {
                if (i + j) % 2 != 0 {
                    continue;
                }
                let c = sh.c + sh.u * (i as f32 * 6.0) + sh.v * (j as f32 * 6.0);
                if fits(c, 1.2, 0.9, &placed) && m.fine_ok() {
                    m.boxm(cx, FINE, c, sh.u * 1.2, sh.v * 0.9, top - 0.05, top + 0.35, kcol([150, 152, 150], K_PLAIN), sty(P_SKYLIGHT), Some(kcol([60, 64, 66], K_PLAIN)));
                }
            }
        }
    }
}

/// Overhanging eaves round a pitched roof: the roof planes carried on past
/// the walls, with a fascia board along the edge.
#[allow(clippy::too_many_arguments)]
fn eaves(cx: &Ctx, m: &mut Mesh, sh: &Shape, top: f32, roof: &Roof, creases: &[(f32, f32, f32)], col: [u8; 4], sty: [u8; 4], fascia: [u8; 4], over: f32) {
    let ring = &sh.rings[0];
    let Some(out) = inset(ring, -over) else { return };
    let n = ring.len();
    let upv = (cx.pos_m(sh.c, top) + cx.center.as_vec3()).normalize();
    let hr = |p: Vec2| {
        let (u, v) = sh.uv(p);
        top + roof.h_raw(u, v, sh.hl, sh.hw)
    };
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b, oa, ob) = (ring[i], ring[j], out[i], out[j]);
        if (b - a).length() < 0.05 {
            continue;
        }
        let (ua, ub) = (sh.uv(a), sh.uv(b));
        let mut ts = vec![0.0f32, 1.0];
        for c in creases {
            let (sa, sb) = (c.0 * ua.0 + c.1 * ua.1 - c.2, c.0 * ub.0 + c.1 * ub.1 - c.2);
            if sa * sb < 0.0 {
                ts.push(sa / (sa - sb));
            }
        }
        ts.sort_by(|x, y| x.total_cmp(y));
        for w in ts.windows(2) {
            if w[1] - w[0] < 1e-4 {
                continue;
            }
            let (p0, p1) = (a + (b - a) * w[0], a + (b - a) * w[1]);
            let (q0, q1) = (oa + (ob - oa) * w[0], ob + (oa - ob) * (1.0 - w[1]));
            let (h0, h1) = (top + { let (u, v) = sh.uv(p0); roof.h(u, v, sh.hl, sh.hw) }, top + { let (u, v) = sh.uv(p1); roof.h(u, v, sh.hl, sh.hw) });
            let (g0, g1) = (hr(q0).min(h0), hr(q1).min(h1));
            let l = (p1 - p0).length();
            m.poly(true, &[(cx.pos_m(p0, h0), [0.0, 0.0]), (cx.pos_m(p1, h1), [l, 0.0]), (cx.pos_m(q1, g1), [l, over]), (cx.pos_m(q0, g0), [0.0, over])], col, sty, Some(upv));
            m.poly(true, &[(cx.pos_m(q0, g0 - 0.22), [0.0, 0.0]), (cx.pos_m(q1, g1 - 0.22), [l, 0.0]), (cx.pos_m(q1, g1), [l, 0.22]), (cx.pos_m(q0, g0), [0.0, 0.22])], fascia, [0, P_PLAIN, 0, 0], None);
        }
    }
}

/// Things on a pitched roof seen close up: dormers, roof windows, and solar
/// panels on the slope that faces the sun.
#[allow(clippy::too_many_arguments)]
fn roof_dressing(cx: &Ctx, m: &mut Mesh, sh: &Shape, top: f32, st: &Style, seed: u32) {
    if !m.fine_ok() {
        return;
    }
    let (hl, hw) = (sh.hl, sh.hw);
    let (off, ridge) = match st.roof {
        Roof::Gable(r, off) => (off, r),
        Roof::Hip(p) if hl > hw * 1.3 => (0.0, p * hw),
        _ => return,
    };
    if hl < 3.0 || hw < 2.5 {
        return;
    }
    let w8 = (st.wealth * 255.0) as u8;
    let eq = Vec2::new(0.0, if cx.lat >= 0.0 { 1.0 } else { -1.0 });
    // Which side of the ridge faces the sun: the v > off plane slopes down toward +v.
    let sunny = if sh.v.dot(eq) >= 0.0 { 1.0 } else { -1.0 };
    let plane = |u: f32, v: f32, lift: f32| {
        let p = sh.c + sh.u * u + sh.v * v;
        cx.pos_m(p, top + st.roof.h(u, v, hl, hw) + lift)
    };
    let span = |side: f32| if side > 0.0 { (off, hw) } else { (-hw, off) };
    let upv = (cx.pos_m(sh.c, top) + cx.center.as_vec3()).normalize();
    // Solar panels: common where there is money and sun.
    let solar_p = match cx.region {
        Region::NAm | Region::Oceania => 0.08 + 0.35 * st.wealth,
        Region::NEur | Region::Med => 0.12 + 0.3 * st.wealth,
        Region::EAsia => 0.1,
        _ => 0.04,
    };
    let (v0, v1) = span(sunny);
    let hu = if matches!(st.roof, Roof::Hip(_)) { hl - hw * 0.9 } else { hl - 0.8 };
    if hash(seed, 90) < solar_p && hu > 1.5 {
        let (a, b) = (v0 + (v1 - v0) * 0.15, v0 + (v1 - v0) * 0.8);
        let w = hu.min(5.0 + hash(seed, 91) * 4.0);
        let u0 = (hash(seed, 92) - 0.5) * (hu - w);
        let pts = [(u0 - w, a), (u0 + w, a), (u0 + w, b), (u0 - w, b)];
        let lp: Vec<(Vec3, [f32; 2])> = pts.iter().map(|(u, v)| (plane(*u, *v, 0.09), [*u - (u0 - w), (*v - a).abs() * 1.25])).collect();
        m.poly(FINE, &lp, kcol([20, 26, 50], K_PLAIN), [0, P_SOLAR, w8, 0], Some(upv));
    } else if hash(seed, 93) < 0.3 + 0.3 * st.wealth {
        // Roof windows.
        for k in 0..1 + (hash(seed, 94) * 2.5) as u32 {
            let side = if hash(seed, 95 + k) < 0.5 { 1.0 } else { -1.0 };
            let (s0, s1) = span(side);
            let u = (hash(seed, 98 + k) - 0.5) * hu * 1.6;
            let v = s0 + (s1 - s0) * 0.45;
            let (du, dv) = (0.4, (s1 - s0).abs() * 0.12);
            let pts = [(u - du, v - dv), (u + du, v - dv), (u + du, v + dv), (u - du, v + dv)];
            let lp: Vec<(Vec3, [f32; 2])> = pts.iter().map(|(a, b)| (plane(*a, *b, 0.07), [*a - (u - du), *b - (v - dv)])).collect();
            m.poly(FINE, &lp, kcol([90, 92, 94], K_PLAIN), [(hash(seed, 99) * 255.0) as u8, P_SKYLIGHT, w8, 0], Some(upv));
        }
    }
    // Dormers in the steep roofs of northern Europe and old America.
    if matches!(cx.region, Region::NEur | Region::NAm) && ridge > 2.2 && hash(seed, 110) < 0.4 + 0.3 * st.wealth {
        let side = -sunny;
        let (s0, s1) = span(side);
        let nd = ((hu * 2.0 / 3.4) as i32).clamp(1, 3);
        for k in 0..nd {
            let u = (k as f32 + 0.5) / nd as f32 * hu * 1.6 - hu * 0.8;
            let vf = s0 + (s1 - s0) * if side > 0.0 { 0.35 } else { 0.65 };
            let ve = s0 + (s1 - s0) * if side > 0.0 { 0.85 } else { 0.15 };
            let (zb, zt) = (top + st.roof.h(u, ve, hl, hw) - 0.1, top + st.roof.h(u, vf, hl, hw));
            if zt - zb < 0.9 {
                continue;
            }
            let c = sh.c + sh.u * u + sh.v * ((vf + ve) * 0.5);
            let half_d = (ve - vf).abs() * 0.5;
            m.boxm(cx, FINE, c, sh.u * 0.75, sh.v * half_d, zb, zt, kcol(st.wall, K_WALL), [(hash(seed, 111 + k as u32) * 255.0) as u8, 0, w8, (zt - zb) as u8], Some(kcol(mixc(st.roof_col, [60, 60, 60], 0.3), K_PLAIN)));
            // The dormer window, on the face looking down the slope.
            let face = sh.c + sh.u * u + sh.v * ve;
            let out = sh.v * side * 0.02;
            let w = 0.45;
            let wp = [(face + out - sh.u * w, zb + 0.35), (face + out + sh.u * w, zb + 0.35), (face + out + sh.u * w, zt - 0.2), (face + out - sh.u * w, zt - 0.2)];
            let lp: Vec<(Vec3, [f32; 2])> = wp.iter().enumerate().map(|(i, (p, z))| (cx.pos_m(*p, *z), [if i == 1 || i == 2 { 0.9 } else { 0.0 }, *z - zb])).collect();
            m.poly(FINE, &lp, kcol([40, 44, 48], K_PLAIN), [(hash(seed, 115 + k as u32) * 255.0) as u8, P_SKYLIGHT, w8, 0], None);
        }
    }
}

/// Things hung on the walls, matching the window grid the shader draws:
/// air conditioners under apartment windows, awnings over shop fronts.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn wall_dressing(cx: &Ctx, m: &mut Mesh, sh: &Shape, st: &Style, seed8: u8, seed: u32, base: f32, top: f32) {
    let ac_p = match cx.region {
        Region::SAsia | Region::EAsia | Region::SeAsia | Region::Mena => 0.35,
        Region::LatAm | Region::Med | Region::Africa => 0.22,
        Region::NAm => 0.12 * (1.0 - st.wealth),
        _ => 0.0,
    };
    let shop = st.facade == 5;
    if !matches!(st.facade, 1 | 5 | 6) || !m.fine_ok() || (ac_p <= 0.0 && !shop) {
        return;
    }
    let sd = seed8 as f32 / 255.0;
    let (cell_w, floor_h) = match st.facade {
        1 => (2.5 + sd * 0.9, 3.2),
        5 => (2.8 + sd, 3.6),
        _ => (3.2 + sd * 0.6, 3.0),
    };
    let ring = &sh.rings[0];
    let n = ring.len();
    let area: f32 = (0..n).map(|i| ring[i].perp_dot(ring[(i + 1) % n])).sum();
    let floors = ((top - sh.gmin) / floor_h) as i32;
    let mut along = 0.0;
    let mut budget = 48;
    let mut awn = 0;
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let l = (b - a).length();
        if l < 0.05 {
            continue;
        }
        let d = (b - a) / l;
        // Outward: right of travel for a counter-clockwise ring.
        let out = if area > 0.0 { -d.perp() } else { d.perp() };
        if l > 4.0 {
            // Window centres: where the shader's bay grid puts them.
            let k0 = ((along / cell_w + sd * 7.0) - 0.5).ceil() as i32;
            let mut k = k0;
            loop {
                let x = (k as f32 + 0.5 - sd * 7.0) * cell_w - along;
                if x > l - 1.0 {
                    break;
                }
                k += 1;
                if x < 1.0 {
                    continue;
                }
                let p = a + d * x;
                for fl in 1..floors.min(30) {
                    if budget == 0 || !m.fine_ok() {
                        break;
                    }
                    let h = hash(seed ^ (k as u32).wrapping_mul(7919), 200 + fl as u32);
                    let z = sh.gmin + fl as f32 * floor_h + 0.35;
                    // Only on this part's own stretch of wall.
                    if h < ac_p && z > base + 0.3 && z + 0.5 < top {
                        m.boxm(cx, FINE, p + out * 0.27, d * 0.36, out * 0.26, z, z + 0.48, kcol([200, 200, 196], K_PLAIN), [seed8, P_AC, 0, 0], None);
                        budget -= 1;
                    }
                }
            }
        }
        // Shop awnings along the ground floor.
        if shop && base <= sh.gmin + 0.5 && l > 3.0 && awn < 12 {
            // Shop windows sit every 2.6 m along the wall (see the shader).
            let mut x = (((along + 1.3) / 2.6 - 0.5).ceil() + 0.5) * 2.6 - along;
            while x < l - 1.3 && awn < 12 {
                if hash(seed, 300 + (along + x) as u32) < 0.5 {
                    let p = a + d * x;
                    let col = pick(&[[150, 30, 30], [30, 70, 40], [30, 50, 110], [190, 150, 40], [80, 40, 30], [40, 40, 44]], hash(seed, 400 + (along + x) as u32));
                    let (z0, z1) = (sh.gmin + 3.15, sh.gmin + 2.55);
                    let (w0, w1) = (p - d * 1.15, p + d * 1.15);
                    let q = |p: Vec2, o: f32, z: f32| (cx.pos_m(p + out * o, z), [0.0f32, 0.0f32]);
                    let mut quad = [q(w0, 0.05, z0), q(w1, 0.05, z0), q(w1, 1.3, z1), q(w0, 1.3, z1)];
                    quad[1].1 = [2.3, 0.0];
                    quad[2].1 = [2.3, 1.3];
                    quad[3].1 = [0.0, 1.3];
                    m.poly(FINE, &quad, kcol(col, K_PLAIN), [seed8, P_AWNING, 0, 0], None);
                    let val = [q(w0, 1.3, z1), q(w1, 1.3, z1), q(w1, 1.3, z1 - 0.3), q(w0, 1.3, z1 - 0.3)];
                    m.poly(FINE, &val, kcol(col, K_PLAIN), [seed8, P_AWNING, 0, 0], None);
                    awn += 1;
                }
                x += 2.6;
            }
        }
        along += l;
    }
}

/// How the footprints of a tile relate to each other. OpenStreetMap often
/// models one building as several parts - floor slabs stacked by their
/// minimum height, a tower rising out of a podium, a spire through the
/// middle - and some footprints are simply mapped twice. Built one by one,
/// every part became a building of its own, with its own colour, facade and
/// rooftop plant, so one tower read as ten buildings jammed into each other.
struct Rel {
    keep: bool,
    /// metres above the ground the part spans, after merging and trimming
    lo: f32,
    hi: f32,
    /// the part whose footprint and look the whole structure takes (itself when alone)
    root: usize,
    /// one of several parts of a structure
    multi: bool,
    /// height of the whole structure
    top: f32,
    /// share of the roof that another part sits on or rises through
    capped: f32,
    /// parts that sit on or rise through the roof: rooftop plant keeps off them
    above: Vec<usize>,
}

fn inside_poly(p: [f32; 2], b: &crate::sources::BuildingPoly) -> bool {
    point_in_ring(p, &b.rings[0]) && !b.rings[1..].iter().any(|h| point_in_ring(p, h))
}

fn relate(unit_m: f32, bs: &[crate::sources::BuildingPoly]) -> Vec<Rel> {
    let n = bs.len();
    let area: Vec<f32> = bs.iter().map(|b| crate::mvt::signed_area(&b.rings[0]).abs() * unit_m * unit_m).collect();
    let mut lo: Vec<f32> = bs.iter().map(|b| if b.min_height > 0.5 { b.min_height } else { 0.0 }).collect();
    let mut hi: Vec<f32> = bs.iter().zip(&lo).map(|(b, l)| b.height.max(2.5).max(l + 0.5)).collect();
    let mut keep: Vec<bool> = area.iter().map(|a| *a >= 4.0).collect();

    // Each footprint is stood for by a grid of points inside it; how much of
    // one lies inside another is the share of its points that do.
    let mut bbox = vec![([0f32; 2], [0f32; 2]); n];
    let mut pts: Vec<Vec<[f32; 2]>> = vec![Vec::new(); n];
    for (i, b) in bs.iter().enumerate() {
        let (mut a, mut z) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in &b.rings[0] {
            a = [a[0].min(p[0]), a[1].min(p[1])];
            z = [z[0].max(p[0]), z[1].max(p[1])];
        }
        bbox[i] = (a, z);
        const G: usize = 10;
        for y in 0..G {
            for x in 0..G {
                let p = [a[0] + (z[0] - a[0]) * (x as f32 + 0.5) / G as f32, a[1] + (z[1] - a[1]) * (y as f32 + 0.5) / G as f32];
                if inside_poly(p, b) {
                    pts[i].push(p);
                }
            }
        }
        if pts[i].is_empty() {
            pts[i].push([(a[0] + z[0]) * 0.5, (a[1] + z[1]) * 0.5]);
        }
    }
    let share = |i: usize, j: usize| pts[i].iter().filter(|p| inside_poly(**p, &bs[j])).count() as f32 / pts[i].len() as f32;
    // nb[i]: (j, share of i inside j) for every footprint touching i.
    let mut nb: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|a, b| bbox[*a].0[0].total_cmp(&bbox[*b].0[0]));
    for (k, &i) in ord.iter().enumerate() {
        for &j in &ord[k + 1..] {
            if bbox[j].0[0] > bbox[i].1[0] {
                break;
            }
            if bbox[j].0[1] > bbox[i].1[1] || bbox[j].1[1] < bbox[i].0[1] {
                continue;
            }
            let (fij, fji) = (share(i, j), share(j, i));
            if fij > 0.0 || fji > 0.0 {
                nb[i].push((j, fij));
                nb[j].push((i, fji));
            }
        }
    }
    let f = |i: usize, j: usize| nb[i].iter().find(|e| e.0 == j).map_or(0.0, |e| e.1);

    // The same footprint mapped twice, and slabs of one footprint and colour
    // stacked straight on top of each other, become one volume.
    let mut by_lo: Vec<usize> = (0..n).collect();
    by_lo.sort_by(|a, b| lo[*a].total_cmp(&lo[*b]));
    for &i in &by_lo {
        while keep[i] {
            let mut grew = false;
            for &(j, fij) in &nb[i] {
                let twin = fij >= 0.85 && f(j, i) >= 0.85;
                let stacked = bs[i].colour == bs[j].colour && lo[j] >= lo[i] - 0.01 && lo[j] <= hi[i] + 0.5;
                if keep[j] && twin && ((lo[j] - lo[i]).abs() <= 0.5 || stacked) {
                    hi[i] = hi[i].max(hi[j]);
                    keep[j] = false;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
    }
    // A part that lies entirely within another one is never seen.
    let mut by_vol: Vec<usize> = (0..n).collect();
    by_vol.sort_by(|a, b| (area[*b] * (hi[*b] - lo[*b])).total_cmp(&(area[*a] * (hi[*a] - lo[*a]))).then(a.cmp(b)));
    let mut rank = vec![0; n];
    for (k, &i) in by_vol.iter().enumerate() {
        rank[i] = k;
    }
    for &i in &by_vol {
        if keep[i] && nb[i].iter().any(|&(j, fij)| keep[j] && rank[j] < rank[i] && fij >= 0.85 && lo[j] <= lo[i] + 0.5 && hi[j] >= hi[i] - 0.5) {
            keep[i] = false;
        }
    }
    // A part sitting on another one but starting a little below its roof: the
    // lower one stops where the upper one starts when the upper covers it,
    // otherwise the upper one starts on the lower one's roof. Their walls then
    // never share the same band of facade.
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for &(j, fij) in &nb[i] {
            if keep[j] && lo[j] > lo[i] + 0.5 && lo[j] < hi[i] && hi[j] > hi[i] + 0.5 {
                if fij >= 0.9 {
                    hi[i] = lo[j];
                } else if f(j, i) >= 0.9 {
                    lo[j] = hi[i];
                }
            }
        }
    }

    // Structures: parts that overlap a good deal and touch in height. A tiny
    // footprint inside a big one (a tower in a mall) keeps its own look.
    let mut up: Vec<usize> = (0..n).collect();
    fn find(up: &mut [usize], mut i: usize) -> usize {
        while up[i] != i {
            up[i] = up[up[i]];
            i = up[i];
        }
        i
    }
    for i in 0..n {
        for &(j, fij) in &nb[i] {
            if j < i || !keep[i] || !keep[j] {
                continue;
            }
            let s = if area[i] < area[j] { fij } else { f(j, i) };
            let ratio = area[i].min(area[j]) / area[i].max(area[j]);
            if s >= 0.3 && ratio >= 0.1 && lo[j] <= hi[i] + 1.5 && lo[i] <= hi[j] + 1.5 {
                let (a, b) = (find(&mut up, i), find(&mut up, j));
                up[a] = b;
            }
        }
    }
    // Root: the biggest footprint standing on the ground.
    let mut best: Vec<Option<usize>> = vec![None; n];
    let mut count = vec![0; n];
    let mut top = vec![0f32; n];
    let better = |a: usize, b: usize| ((lo[a] <= 0.5) as u8, area[a]) > ((lo[b] <= 0.5) as u8, area[b]);
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let g = find(&mut up, i);
        count[g] += 1;
        top[g] = top[g].max(hi[i]);
        if best[g].is_none_or(|b| better(i, b)) {
            best[g] = Some(i);
        }
    }

    (0..n)
        .map(|i| {
            let g = find(&mut up, i);
            let above: Vec<usize> = if keep[i] { nb[i].iter().filter(|&&(j, fij)| keep[j] && fij > 0.0 && lo[j] <= hi[i] + 1.0 && hi[j] > hi[i] + 0.3).map(|e| e.0).collect() } else { Vec::new() };
            let capped = if above.is_empty() { 0.0 } else { pts[i].iter().filter(|p| above.iter().any(|&j| inside_poly(**p, &bs[j]))).count() as f32 / pts[i].len() as f32 };
            Rel { keep: keep[i], lo: lo[i], hi: hi[i], root: best[g].unwrap_or(i), multi: keep[i] && count[g] > 1, top: top[g].max(hi[i]), capped, above }
        })
        .collect()
}

/// Footprint rings without closing points, the vertex centre and ground heights under the outline.
fn site(cx: &Ctx, b: &crate::sources::BuildingPoly) -> Option<(Vec<Vec<[f32; 2]>>, [f32; 2], f32, f32)> {
    let mut rings: Vec<Vec<[f32; 2]>> = b.rings.clone();
    for r in rings.iter_mut() {
        if r.len() > 3 && r.first() == r.last() {
            r.pop();
        }
    }
    rings.retain(|r| r.len() >= 3);
    let outer = rings.first()?;
    let n = outer.len() as f32;
    let cen = outer.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n]);
    let (mut gmin, mut gmax) = (f32::MAX, f32::MIN);
    for p in outer {
        let g = cx.ground(*p);
        gmin = gmin.min(g);
        gmax = gmax.max(g);
    }
    Some((rings, cen, gmin, gmax))
}

/// The look a footprint gets on its own: what `style` decides from what is known about it.
fn own_style(cx: &Ctx, b: &crate::sources::BuildingPoly, rings: &[Vec<[f32; 2]>], cen: [f32; 2], bi: usize, hoods: &Hoods, landuse: &[u8]) -> (Style, [u8; 3], u32) {
    let outer = &rings[0];
    let area = crate::mvt::signed_area(outer).abs() * cx.unit_m * cx.unit_m;
    let rm: Vec<Vec2> = outer.iter().map(|p| cx.metres(*p)).collect();
    let (_, _, hl, hw) = obb(&rm);
    let rect = area / (4.0 * hl * hw);
    let seed = (bi as u32).wrapping_mul(2654435761) ^ cx.id.x.wrapping_mul(97) ^ cx.id.y.wrapping_mul(131);
    let hood = hoods.at(cx, cen);
    let lu = landuse[((cen[1] / cx.ext * 64.0) as usize).min(63) * 64 + ((cen[0] / cx.ext * 64.0) as usize).min(63)];
    let img = roof_colour(cx, outer, cen);
    (style(cx, b, area, hl, hw, rect, outer.len(), hood, lu, img, seed), img, seed)
}

#[allow(clippy::too_many_arguments)]
fn building(cx: &Ctx, m: &mut Mesh, all: &[crate::sources::BuildingPoly], rels: &[Rel], bi: usize, hoods: &Hoods, landuse: &[u8]) -> Option<Footprint> {
    let rel = &rels[bi];
    if !rel.keep {
        return None;
    }
    let src = &all[bi];
    let b = &crate::sources::BuildingPoly { rings: Vec::new(), height: rel.hi, min_height: if rel.lo > 0.5 { rel.lo } else { src.min_height }, colour: src.colour };
    let (rings_u, cen, gmin, gmax) = site(cx, src)?;
    let outer = &rings_u[0];
    // Tiles overlap by a buffer; keep a building only in the tile that owns its centre.
    if cen[0] < 0.0 || cen[1] < 0.0 || cen[0] >= cx.ext || cen[1] >= cx.ext {
        return None;
    }
    let area = crate::mvt::signed_area(outer).abs() * cx.unit_m * cx.unit_m;
    if area < 4.0 {
        return None;
    }
    let rings: Vec<Vec<Vec2>> = rings_u.iter().map(|r| r.iter().map(|p| cx.metres(*p)).collect()).collect();
    let (c, u, hl, hw) = obb(&rings[0]);
    let rect = area / (4.0 * hl * hw);
    // Parts of one structure share its ground level (so floors line up), its
    // facade, colour and upkeep: the look of the whole, from its main footprint.
    let (st, seed, gref, gtop) = if rel.multi {
        let rb = &all[rel.root];
        let (rr, rcen, rgmin, rgmax) = site(cx, rb)?;
        let whole = crate::sources::BuildingPoly { rings: Vec::new(), height: rel.top, min_height: rb.min_height, colour: rb.colour };
        let (mut st, _, rseed) = own_style(cx, &whole, &rr, rcen, rel.root, hoods, landuse);
        if let Some(c) = src.colour {
            st.wall = c;
        }
        // A tagged wall colour is a painted or clad wall, not a curtain of glass that would hide it.
        if st.facade == 3 && (src.colour.is_some() || rb.colour.is_some()) {
            st.facade = 2;
        }
        // Only the main footprint, with nothing on it, keeps a pitched roof.
        if !(bi == rel.root && rel.capped < 0.05) && !matches!(st.roof, Roof::Flat) {
            st.roof = Roof::Flat;
            st.parapet = 0.4;
        }
        // Each roof takes its covering from its own patch of the image.
        let img = roof_colour(cx, outer, cen);
        if matches!(st.roof, Roof::Flat) {
            let lum = (img[0] as f32 + img[1] as f32 + img[2] as f32) / 3.0;
            let (fm, fc) = flat_stuff(st.kind, lum, hash(rseed, 10));
            st.flat_mat = fm;
            st.roof_col = mixc(fc, img, 0.55);
        }
        (st, rseed, rgmin, rgmax.min(rgmin + 3.0))
    } else {
        let (st, _, seed) = own_style(cx, b, &rings_u, cen, bi, hoods, landuse);
        (st, seed, gmin, gmax.min(gmin + 3.0))
    };
    let mut st = st;
    // Roofs mostly covered by the part above carry no parapet or plant.
    let hidden_roof = rel.capped >= 0.6;
    if hidden_roof {
        st.parapet = 0.0;
    }
    let sh = Shape { rings, c, u, v: u.perp(), hl, hw, gmin: gref };
    let avoid: Vec<Vec<Vec2>> = rel.above.iter().map(|&j| all[j].rings[0].iter().map(|p| cx.metres(*p)).collect()).collect();

    let height = b.height.max(2.5);
    let base = if b.min_height > 0.5 { gref + b.min_height } else { gmin - 2.0 };
    let mut top = gtop + height;
    let rmax = st.roof.max_h(hl, hw);
    // The tagged height is to the ridge: pitched roofs take their rise out of the walls.
    if rmax > 0.0 {
        top -= (rmax * 0.6).min(height * 0.45);
    }
    let (i_main, i_det, i_fine, i_far) = (m.main.len() as u32, m.det.len() as u32, m.fine.len() as u32, m.far.len() as u32);
    let seed8 = (hash(seed, 0) * 255.0) as u8;
    let w8 = (st.wealth * 255.0) as u8;
    let h8 = (top + rmax - gref).clamp(0.0, 255.0) as u8;
    let wsty = [seed8, st.facade, w8, h8];
    let psty = [seed8, P_PLAIN, w8, 0];
    let wall = kcol(st.wall, K_WALL);
    let creases = st.roof.creases(hl, hw);
    let flat = matches!(st.roof, Roof::Flat);
    let helipad = flat && !hidden_roof && rel.above.is_empty() && st.kind == Kind::Tower && height > 100.0 && hl > 12.0 && hw > 12.0 && hash(seed, 23) < 0.15;
    let hp = if helipad { HELIPAD } else { 0 };

    // Towers step back near the top, when the map does not already model that with parts.
    let mut tiers: Vec<(Vec<Vec2>, f32)> = Vec::new();
    let mut tier_steps: Vec<(f32, f32)> = Vec::new();
    if st.kind == Kind::Tower && !rel.multi && rel.above.is_empty() && sh.rings.len() == 1 && rect > 0.6 && hash(seed, 20) < 0.6 {
        let mut ring = sh.rings[0].clone();
        let mut z = base;
        let steps = 1 + (hash(seed, 21) * 2.2) as usize;
        for k in 0..steps {
            let frac = 0.62 + 0.3 * (k as f32 + hash(seed, 22 + k as u32)) / steps as f32;
            let zt = gmin + (top - gmin) * frac.min(0.94);
            let d = hw * (0.1 + 0.14 * hash(seed, 26 + k as u32));
            let Some(next) = inset(&ring, d) else { break };
            tiers.push((ring, zt));
            tier_steps.push((d, zt));
            ring = next;
            z = zt;
        }
        let _ = z;
        tiers.push((ring, top));
    }

    if tiers.len() > 1 {
        let mut z0 = base;
        for (k, (ring, zt)) in tiers.iter().enumerate() {
            walls(cx, m, Lvl(0), &sh, ring, z0, *zt, &Roof::Flat, &[], wall, wsty, 0.0);
            // Terrace left by the setback, or the top roof.
            let last = k + 1 == tiers.len();
            let rs = if !last { vec![ring.clone(), tiers[k + 1].0.clone()] } else { vec![ring.clone()] };
            roof_surface(cx, m, Lvl(0), &sh, &rs, *zt, &Roof::Flat, &[], kcol(st.roof_col, K_FLAT), [seed8, st.flat_mat | if last { hp } else { 0 }, w8, 0]);
            z0 = *zt - 0.2;
        }
        let last = &tiers[tiers.len() - 1].0;
        roof_clutter(cx, m, &sh, last, top, &st, seed, height, helipad, &avoid);
    } else {
        for r in &sh.rings {
            walls(cx, m, Lvl(0), &sh, r, base, top, &st.roof, &creases, wall, wsty, if flat { st.parapet } else { 0.0 });
        }
        // Pitched roofs carry their ridge height (decimetres) so the shader can cap the ridge.
        let ridge8 = (rmax * 10.0).clamp(0.0, 255.0) as u8;
        let (rk, rsty) = if flat { (K_FLAT, [seed8, st.flat_mat | hp, w8, 0]) } else { (K_PITCH, [seed8, st.roof_mat, w8, ridge8]) };
        roof_surface(cx, m, Lvl(0), &sh, &sh.rings, top, &st.roof, &creases, kcol(st.roof_col, rk), rsty);
        if flat && st.parapet > 0.0 && !hidden_roof {
            // Parapet: a low wall around the roof edge with an inner face and a cap.
            if let Some(inner) = inset(&sh.rings[0], 0.28) {
                let o = &sh.rings[0];
                let pc = kcol(mixc(st.wall, [130, 130, 128], 0.35), K_PLAIN);
                let upv = (cx.pos_m(sh.c, top) + cx.center.as_vec3()).normalize();
                for i in 0..o.len() {
                    let j = (i + 1) % o.len();
                    let zt = top + st.parapet;
                    let (a, bq, ia, ib) = (cx.pos_m(o[i], zt), cx.pos_m(o[j], zt), cx.pos_m(inner[i], zt), cx.pos_m(inner[j], zt));
                    m.poly(true, &[(a, [0.0, 0.0]), (bq, [1.0, 0.0]), (ib, [1.0, 0.3]), (ia, [0.0, 0.3])], pc, psty, Some(upv));
                    m.poly(true, &[(ia, [0.0, 0.0]), (ib, [1.0, 0.0]), (cx.pos_m(inner[j], top), [1.0, 1.0]), (cx.pos_m(inner[i], top), [0.0, 1.0])], pc, psty, None);
                }
            }
            roof_clutter(cx, m, &sh, &sh.rings[0], top, &st, seed, height, helipad, &avoid);
        }
        if !flat && matches!(st.kind, Kind::House | Kind::Row | Kind::Shed) && sh.rings.len() == 1 {
            let over = if st.kind == Kind::Shed { 0.25 } else { 0.35 + 0.3 * hash(seed, 24) };
            let fascia = kcol(mixc(st.wall, [236, 234, 228], 0.55), K_PLAIN);
            eaves(cx, m, &sh, top, &st.roof, &creases, kcol(st.roof_col, K_PITCH), [seed8, st.roof_mat, w8, 0], fascia, over);
            roof_dressing(cx, m, &sh, top, &st, seed);
        }
        if !flat && matches!(st.kind, Kind::House | Kind::Row) {
            // Chimneys: more of them, and tidier, where there is money.
            let nch = if matches!(cx.region, Region::Mena | Region::SeAsia | Region::Africa | Region::SAsia) { 0 } else { (hash(seed, 40) < 0.35 + 0.45 * st.wealth) as u32 + (st.wealth > 0.7 && hash(seed, 41) < 0.4) as u32 };
            for k in 0..nch {
                let (cu, cv) = ((hash(seed, 42 + k) - 0.5) * hl * 1.3, (hash(seed, 44 + k) - 0.5) * hw * 0.7);
                let p = sh.c + sh.u * cu + sh.v * cv;
                if in_ring_m(p, &sh.rings[0]) {
                    let z = top + st.roof.h(cu, cv, hl, hw);
                    m.boxm(cx, true, p, sh.u * 0.32, sh.v * 0.42, z - 0.6, top + rmax + 0.7, kcol(mixc([140, 82, 64], st.wall, 0.4), K_PLAIN), psty, Some(kcol([40, 36, 34], K_PLAIN)));
                }
            }
        } else if st.kind == Kind::Industrial && !flat {
            // Ridge vents.
            for k in 0..(hl / 9.0) as u32 {
                let cu = (hash(seed, 50 + k) - 0.5) * hl * 1.6;
                let p = sh.c + sh.u * cu;
                if in_ring_m(p, &sh.rings[0]) {
                    let z = top + st.roof.h(cu, 0.0, hl, hw);
                    m.boxm(cx, true, p, sh.u * 0.5, sh.v * 0.5, z - 0.3, z + 0.9, kcol([150, 152, 154], K_PLAIN), psty, None);
                }
            }
        }
    }

    wall_dressing(cx, m, &sh, &st, seed8, seed, base, top);
    let far = Far { rect, area, tiers: &tiers, base, top, hidden_roof, wall, wsty, seed8, w8, hp };
    far_version(cx, m, &sh, &st, &far);

    let ring: Vec<[f32; 2]> = outer.iter().map(|p| [p[0] / cx.ext, p[1] / cx.ext]).collect();
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in &ring {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    Some(Footprint {
        min: lo,
        max: hi,
        ring,
        base: if b.min_height > 0.5 { base } else { gmin },
        top: top + rmax + if flat { st.parapet } else { 0.0 },
        kind: 0,
        col: st.wall,
        main: (i_main, m.main.len() as u32 - i_main),
        det: (i_det, m.det.len() as u32 - i_det),
        fine: (i_fine, m.fine.len() as u32 - i_fine),
        far: (i_far, m.far.len() as u32 - i_far),
        alive: true,
        center: cx.pos(cen, gmin),
        area,
        sty: wsty,
        gref,
        roof_col: st.roof_col,
        tiers: if tiers.len() > 1 { tier_steps } else { Vec::new() },
    })
}

/// What `far_version` needs to know about a building.
struct Far<'a> {
    rect: f32,
    area: f32,
    tiers: &'a [(Vec<Vec2>, f32)],
    base: f32,
    top: f32,
    hidden_roof: bool,
    wall: [u8; 4],
    wsty: [u8; 4],
    seed8: u8,
    w8: u8,
    hp: u8,
}

/// The building as it is seen from more than a mile off, where a metre is
/// under half a pixel: walls and a roof over a simplified footprint (its
/// bounding box, if it is nearly rectangular), the roof reduced to at most
/// one ridge, and none of the dressing. A tower keeps its setbacks. Sheds too
/// small to make out are left out.
fn far_version(cx: &Ctx, m: &mut Mesh, sh: &Shape, st: &Style, f: &Far) {
    if f.area < 30.0 && f.top - sh.gmin < 5.0 {
        return;
    }
    const TOL: f32 = 1.5;
    let flat_roof = kcol(st.roof_col, K_FLAT);
    let flat_sty = |hp: u8| [f.seed8, st.flat_mat | hp, f.w8, 0];
    if f.tiers.len() > 1 {
        let mut z0 = f.base;
        for (k, (ring, zt)) in f.tiers.iter().enumerate() {
            let r = simplify_ring(ring, TOL);
            walls(cx, m, FAR, sh, &r, z0, *zt, &Roof::Flat, &[], f.wall, f.wsty, 0.0);
            let hp = if k + 1 == f.tiers.len() { f.hp } else { 0 };
            roof_surface(cx, m, FAR, sh, &[r], *zt, &Roof::Flat, &[], flat_roof, flat_sty(hp));
            z0 = *zt - 0.2;
        }
        return;
    }
    let (hl, hw) = (sh.hl, sh.hw);
    let outer = if f.rect > 0.85 {
        let (c, u, v) = (sh.c, sh.u * hl, sh.v * hw);
        vec![c - u - v, c + u - v, c + u + v, c - u + v]
    } else {
        simplify_ring(&sh.rings[0], TOL)
    };
    // Courtyards big enough to see stay open.
    let mut rings = vec![outer];
    rings.extend(sh.rings[1..].iter().filter(|h| ring_area(h) > 400.0).map(|h| simplify_ring(h, TOL)));
    let roof = match st.roof {
        Roof::Mansard(p, cap) => Roof::Hip(p.min(cap / hw.min(hl).max(0.5))),
        Roof::Multi(_, r) => Roof::Gable(r, 0.0),
        r => r,
    };
    let flat = matches!(roof, Roof::Flat);
    let creases = roof.creases(hl, hw);
    for r in &rings {
        walls(cx, m, FAR, sh, r, f.base, f.top, &roof, &creases, f.wall, f.wsty, if flat { st.parapet } else { 0.0 });
    }
    if f.hidden_roof {
        return;
    }
    let ridge8 = (roof.max_h(hl, hw) * 10.0).clamp(0.0, 255.0) as u8;
    let (col, sty) = if flat { (flat_roof, flat_sty(f.hp)) } else { (kcol(st.roof_col, K_PITCH), [f.seed8, st.roof_mat, f.w8, ridge8]) };
    roof_surface(cx, m, FAR, sh, &rings, f.top, &roof, &creases, col, sty);
}

/// Douglas-Peucker on a closed ring: drops the corners that lie within `tol`
/// metres of the line between the corners kept on either side.
fn simplify_ring(ring: &[Vec2], tol: f32) -> Vec<Vec2> {
    let n = ring.len();
    if n <= 4 {
        return ring.to_vec();
    }
    let far_from = |p: Vec2| (0..n).max_by(|&a, &b| ring[a].distance_squared(p).total_cmp(&ring[b].distance_squared(p))).unwrap_or(0);
    let a = far_from(ring[0]);
    let b = far_from(ring[a]);
    if a == b {
        return ring.to_vec();
    }
    let mut keep = vec![false; n];
    keep[a] = true;
    keep[b] = true;
    let mut todo = vec![(a, b), (b, a)];
    while let Some((i, j)) = todo.pop() {
        let (p, d) = (ring[i], ring[j] - ring[i]);
        let len = d.length();
        let (mut worst, mut at) = (0.0f32, usize::MAX);
        let mut k = (i + 1) % n;
        while k != j {
            let e = ring[k] - p;
            let off = if len < 1e-4 { e.length() } else { e.perp_dot(d).abs() / len };
            if off > worst {
                (worst, at) = (off, k);
            }
            k = (k + 1) % n;
        }
        if worst > tol {
            keep[at] = true;
            todo.push((i, at));
            todo.push((at, j));
        }
    }
    let out: Vec<Vec2> = (0..n).filter(|&k| keep[k]).map(|k| ring[k]).collect();
    if out.len() < 3 { ring.to_vec() } else { out }
}

fn ring_area(r: &[Vec2]) -> f32 {
    (0..r.len()).map(|i| r[i].perp_dot(r[(i + 1) % r.len()])).sum::<f32>().abs() * 0.5
}

/// Load the satellite image for a detail tile: its four children one zoom down.
fn load_atlas(src: &Sources, id: TileId) -> Vec<u8> {
    let mut atlas = vec![110u8; ATLAS * ATLAS * 4];
    for (k, c) in id.children().iter().enumerate() {
        if let Ok(img) = src.imagery(*c) {
            let img = if img.width() as usize != ATLAS / 2 { image::imageops::resize(&img, (ATLAS / 2) as u32, (ATLAS / 2) as u32, image::imageops::FilterType::Triangle) } else { img };
            let (ox, oy) = ((k % 2) * ATLAS / 2, (k / 2) * ATLAS / 2);
            for y in 0..ATLAS / 2 {
                let row = &img.as_raw()[y * ATLAS / 2 * 4..(y + 1) * ATLAS / 2 * 4];
                atlas[((oy + y) * ATLAS + ox) * 4..((oy + y) * ATLAS + ox + ATLAS / 2) * 4].copy_from_slice(row);
            }
        }
    }
    atlas
}

/// Rasterise polygons (tile units) into a w x w grid with `value`.
pub fn raster(buf: &mut [u8], w: usize, ext: f32, rings: &[Vec<[f32; 2]>], value: u8) {
    let k = w as f32 / ext;
    let px: Vec<Vec<[f32; 2]>> = rings.iter().map(|r| r.iter().map(|p| [p[0] * k, p[1] * k]).collect()).collect();
    crate::mvt::fill_rings(buf, w, w, &px, value);
}

pub fn build(src: &Sources, id: TileId) -> Result<BuildingData> {
    debug_assert_eq!(id.z, DATA_ZOOM);
    let vt = src.vector(id)?;
    let (clat, clon) = id.lat_lon(0.5, 0.5);
    let ext = vt.extent;
    let cx = Ctx {
        id,
        ext,
        unit_m: (id.width_m() / ext as f64) as f32,
        center: geo::lla_to_pos(clat, clon, 0.0),
        elev: src.elev(id).ok(),
        atlas: load_atlas(src, id),
        lat: clat.to_degrees(),
        lon: clon.to_degrees(),
        region: region(clat.to_degrees(), clon.to_degrees()),
    };
    let hoods = neighbourhoods(&cx, &vt);
    let mut landuse = vec![0u8; 64 * 64];
    for want in [LU_RESIDENTIAL, LU_COMMERCIAL, LU_INDUSTRIAL, LU_OPEN] {
        for (k, rings) in vt.landuse.iter().filter(|l| l.0 == want) {
            let mut layer = vec![0u8; 64 * 64];
            raster(&mut layer, 64, ext, rings, 1);
            for (d, s) in landuse.iter_mut().zip(&layer) {
                if *s != 0 {
                    *d = *k;
                }
            }
        }
    }

    let mut m = Mesh::default();
    let mut footprints = Vec::new();
    // Tallest first, so the far shadow cascade can draw just the start of the index run.
    let rels = relate(cx.unit_m, &vt.buildings);
    let mut order: Vec<usize> = (0..vt.buildings.len()).collect();
    order.sort_by(|a, b| rels[*b].hi.total_cmp(&rels[*a].hi));
    let mut n_far_tall = 0;
    // Ends of each height class in the four runs (relative to each run's start).
    let mut cls: [Option<[u32; 4]>; 4] = [None; 4];
    let mut built = vec![0u8; RASTER * RASTER];
    for bi in order {
        let b = &vt.buildings[bi];
        if let Some(f) = building(&cx, &mut m, &vt.buildings, &rels, bi, &hoods, &landuse) {
            footprints.push(f);
            raster(&mut built, RASTER, ext, &b.rings[..1], 1);
        }
        if rels[bi].hi >= 12.0 {
            n_far_tall = m.far.len() as u32;
        }
        cls[hclass(rels[bi].hi)] = Some([m.main.len(), m.det.len(), m.fine.len(), m.far.len()].map(|n| n as u32));
    }
    let n_bld = m.main.len() as u32;

    let mut roads = crate::roads::build(&cx, &vt, src, &mut m, &mut footprints, &built);
    let parked = crate::roads::park_cars(&cx, &roads.net, &hoods, &built);
    let grown = crate::trees::place(&cx, &vt, &hoods, &built, &roads.mask);

    // Affine map from tile-local position to atlas uv.
    let (p0, px, py) = (cx.pos([ext * 0.5, ext * 0.5], 0.0), cx.pos([ext, ext * 0.5], 0.0) - cx.pos([0.0, ext * 0.5], 0.0), cx.pos([ext * 0.5, ext], 0.0) - cx.pos([ext * 0.5, 0.0], 0.0));
    let (ax, ay) = (px / px.length_squared(), py / py.length_squared());
    let n_main = m.main.len() as u32;
    let n_det = n_main + m.det.len() as u32;
    let n_far = n_det + m.fine.len() as u32;
    for f in footprints.iter_mut() {
        f.det.0 += n_main;
        f.fine.0 += n_det;
        f.far.0 += n_far;
    }
    // Empty classes end where the one before does; the last takes the rest of
    // each run (street furniture in the middle and finest levels).
    let n_idx = n_far + m.far.len() as u32;
    let starts = [0, n_main, n_det, n_far];
    let mut cls_abs = [[0u32; 4]; 4];
    let mut prev = [0u32; 4];
    for (c, e) in cls.iter().enumerate() {
        let e = if c + 1 == HCLASS.len() { [n_bld, n_det - n_main, n_far - n_det, n_idx - n_far] } else { e.unwrap_or(prev) };
        prev = e;
        for r in 0..4 {
            cls_abs[r][c] = starts[r] + e[r];
        }
    }
    let mut indices = m.main;
    indices.extend(m.det);
    indices.extend(m.fine);
    indices.extend(m.far);
    Ok(BuildingData {
        id,
        center: cx.center,
        radius: m.radius.max(roads.radius) as f64 + 30.0,
        ext: cx.ext,
        unit_m: cx.unit_m,
        hoods,
        landuse,
        verts: m.v,
        indices,
        n_main,
        n_det,
        n_bld,
        n_far,
        n_far_tall: n_far + n_far_tall,
        cls: cls_abs,
        atlas: mip_chain(cx.atlas, ATLAS, 4),
        atlas_size: ATLAS as u32,
        ax: [ax.x, ax.y, ax.z, 0.5 - ax.dot(p0)],
        ay: [ay.x, ay.y, ay.z, 0.5 - ay.dot(p0)],
        footprints,
        road_verts: std::mem::take(&mut roads.verts),
        road_idx: std::mem::take(&mut roads.idx),
        net: roads.net,
        parked,
        trees: grown.trees,
        tree_var: grown.var,
        chunks: grown.chunks,
        under: grown.under,
        cells: grown.cells,
        near_idx: grown.near_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roads::{Path, RoadNet};

    #[test]
    fn hood_and_landuse_sampling() {
        let mut cells = vec![Hood::default(); NB * NB];
        for (i, h) in cells.iter_mut().enumerate() {
            h.wealth = i as f32 / (NB * NB - 1) as f32;
            h.cover = 0.5;
            h.green = 0.25;
            h.mean_h = 18.0;
        }
        let hoods = Hoods(cells);
        let h_lo = hoods.sample_uv(0.0, 0.0);
        let h_hi = hoods.sample_uv(1.0, 1.0);
        let h_mid = hoods.sample_uv(0.5, 0.5);
        assert!(h_lo.wealth < h_mid.wealth && h_mid.wealth < h_hi.wealth);
        assert!((h_mid.cover - 0.5).abs() < 1e-5);

        let mut lu = vec![0u8; 64 * 64];
        lu[16 * 64 + 32] = LU_COMMERCIAL;
        assert_eq!(sample_landuse(&lu, 32.5 / 64.0, 16.5 / 64.0), LU_COMMERCIAL);
        assert_eq!(sample_landuse(&lu, 0.1, 0.1), 0);
    }

    #[test]
    fn circle_ring_wall_collision_en_frame() {
        // 20m x 20m building centred in a 1000m tile (u: 0.49..0.51 east, v: 0.49..0.51 south).
        let tile_m = 1000.0;
        let ring = [[0.49, 0.49], [0.51, 0.49], [0.51, 0.51], [0.49, 0.51]];

        // Approaching from the East (u = 0.511 => 1.0m east of the east wall at u = 0.51) with radius 1.5m.
        let hit_east = collide_ring_circle([0.511, 0.50], 1.5, tile_m, &ring).expect("should hit east wall");
        assert!((hit_east.0.x - 1.0).abs() < 1e-3 && hit_east.0.y.abs() < 1e-3);
        assert!((hit_east.1 - 0.5).abs() < 1e-2);

        // Approaching from the North (v = 0.489 => 1.0m north of the north wall at v = 0.49) with radius 1.5m.
        let hit_north = collide_ring_circle([0.50, 0.489], 1.5, tile_m, &ring).expect("should hit north wall");
        assert!(hit_north.0.x.abs() < 1e-3 && (hit_north.0.y - 1.0).abs() < 1e-3);
        assert!((hit_north.1 - 0.5).abs() < 1e-2);

        // Inside the footprint near the west wall (u = 0.492 => 2.0m inside west wall at u = 0.49) with radius 0.5m.
        let hit_inside = collide_ring_circle([0.492, 0.50], 0.5, tile_m, &ring).expect("inside should push out");
        assert!((hit_inside.0.x + 1.0).abs() < 1e-3 && hit_inside.0.y.abs() < 1e-3);
        assert!((hit_inside.1 - 2.5).abs() < 1e-2);

        // Outside beyond radius.
        assert!(collide_ring_circle([0.52, 0.50], 1.5, tile_m, &ring).is_none());
    }

    #[test]
    fn nearest_road_lane_snap() {
        let lat = 0.7f64;
        let lon = -1.2f64;
        let center = geo::lla_to_pos(lat, lon, 0.0);
        let (east, north, _) = geo::enu(center);
        let e3 = east.as_vec3();
        let path = Path {
            pts: vec![-e3 * 50.0, e3 * 50.0],
            cum: vec![0.0, 100.0],
            len: 100.0,
            class: 2,
            oneway: false,
            lanes: 1,
            lane_w: 3.5,
            park_off: 0.0,
            speed: 13.0,
            hidden: false,
            nodes: [1, 2],
            edge: [0, 0],
            junctions: Vec::new(),
            keep_clear: Vec::new(),
        };
        let net = RoadNet { paths: vec![path], nodes: Default::default(), left: false, total_len: 100.0 };
        // Query from south of the road: should snap to the eastbound (south/right-hand) lane (+East heading).
        let query = center - north * 10.0;
        let (_, snapped, fwd) = nearest_lane_in_net(center, &net, query).expect("lane found");
        assert!((fwd.x - 1.0).abs() < 1e-2 && fwd.y.abs() < 1e-2);
        let south_offset = -(snapped - center).dot(north);
        assert!((south_offset - 1.75).abs() < 0.1);
    }
}
