//! Traffic: procedurally modelled vehicles and a light simulation that drives
//! them along the road graph of each detail tile.
use crate::buildings::hash;
use crate::render::MeshVertex;
use crate::roads::{car_model, paint, near_keys, Link, RoadNet};
use glam::Vec3;

pub const N_MODELS: usize = 8;
pub const CAR_LEN: [f32; N_MODELS] = [4.7, 4.0, 4.7, 5.4, 5.2, 4.4, 11.5, 7.6];

#[derive(Default)]
struct MB {
    v: Vec<MeshVertex>,
    i: Vec<u32>,
}
impl MB {
    fn poly(&mut self, p: &[Vec3], col: [f32; 4]) {
        let mut n = Vec3::ZERO;
        for k in 1..p.len() - 1 {
            n += (p[k] - p[0]).cross(p[k + 1] - p[0]);
        }
        let n = n.normalize_or_zero();
        let b = self.v.len() as u32;
        for q in p {
            self.v.push(MeshVertex { pos: (*q).into(), nrm: n.into(), col });
        }
        for k in 1..p.len() as u32 - 1 {
            self.i.extend([b, b + k, b + k + 1]);
        }
    }
    fn cuboid(&mut self, lo: Vec3, hi: Vec3, col: [f32; 4]) {
        let p = |x: bool, y: bool, z: bool| Vec3::new(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z });
        self.poly(&[p(false, false, true), p(true, false, true), p(true, true, true), p(false, true, true)], col);
        self.poly(&[p(true, false, false), p(false, false, false), p(false, true, false), p(true, true, false)], col);
        self.poly(&[p(true, false, true), p(true, false, false), p(true, true, false), p(true, true, true)], col);
        self.poly(&[p(false, false, false), p(false, false, true), p(false, true, true), p(false, true, false)], col);
        self.poly(&[p(false, true, true), p(true, true, true), p(true, true, false), p(false, true, false)], col);
        self.poly(&[p(false, false, false), p(true, false, false), p(true, false, true), p(false, false, true)], col);
    }
    /// Average normals across faces that meet at a shallow angle, so panels
    /// read as curved sheet metal while creases stay crisp.
    fn smooth(&mut self, cos_limit: f32) {
        let mut map: std::collections::HashMap<[i32; 3], Vec<usize>> = std::collections::HashMap::new();
        for (k, v) in self.v.iter().enumerate() {
            map.entry([(v.pos[0] * 500.0).round() as i32, (v.pos[1] * 500.0).round() as i32, (v.pos[2] * 500.0).round() as i32]).or_default().push(k);
        }
        let orig: Vec<Vec3> = self.v.iter().map(|v| Vec3::from(v.nrm)).collect();
        for group in map.values() {
            for &a in group {
                let mut n = Vec3::ZERO;
                for &b in group {
                    if orig[a].dot(orig[b]) > cos_limit && (self.v[a].col[3] - self.v[b].col[3]).abs() < 0.1 {
                        n += orig[b];
                    }
                }
                self.v[a].nrm = n.normalize_or_zero().into();
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Spec {
    pub len: f32,
    pub w: f32,
    pub h: f32,
    pub clear: f32,
    pub belt: f32,
    pub hood: f32,
    pub deck: f32,
    pub hood_end: f32,
    pub roof0: f32,
    pub roof1: f32,
    pub deck0: f32,
    pub wheel_r: f32,
    pub axles: &'static [f32],
    pub inset: f32,
    pub glass_to: f32,
    pub bed: bool,
    pub flat_front: bool,
    pub cargo: bool,
}

fn lerp_knots(k: &[(f32, f32)], f: f32) -> f32 {
    for w in k.windows(2) {
        if f <= w[1].0 {
            let t = ((f - w[0].0) / (w[1].0 - w[0].0).max(1e-5)).clamp(0.0, 1.0);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    k[k.len() - 1].1
}

fn vehicle(s: &Spec) -> (Vec<MeshVertex>, Vec<u32>) {
    const PAINT: f32 = 0.0;
    let paint = [1.0, 1.0, 1.0, PAINT];
    let glass = [0.02, 0.03, 0.04, 0.5];
    let trim = [0.025, 0.025, 0.028, 1.0];
    let tyre = [0.012, 0.012, 0.013, 1.0];
    let mut m = MB::default();
    let top = [(0.0, s.hood - 0.14), (0.04, s.hood - 0.03), (s.hood_end, s.hood + 0.04), (s.roof0, s.h), (s.roof1, s.h - 0.02), (s.deck0, s.deck), (0.97f32.max(s.deck0), s.deck - 0.03), (1.0, s.deck - 0.1)];
    let wide = [(0.0, s.w * 0.80), (0.035, s.w * 0.94), (0.12, s.w), (0.9, s.w), (0.97, s.w * 0.96), (1.0, s.w * 0.84)];
    let mut fs = vec![0.0, 0.035, 0.12, s.hood_end, s.roof0, (s.roof0 + s.roof1) * 0.5, s.roof1, s.deck0, 0.9, 0.97, 1.0];
    fs.sort_by(|a, b| a.total_cmp(b));
    fs.dedup_by(|a, b| (*a - *b).abs() < 0.008);
    // Half cross-section at each station, bottom to roof centre.
    let ring = |f: f32| -> [Vec3; 5] {
        let z = (f - 0.5) * s.len;
        let wb = lerp_knots(&wide, f);
        let t = lerp_knots(&top, f);
        let cabin = f > s.hood_end + 0.001 && f < s.deck0 - 0.001;
        let ys = (s.belt + 0.06 * (f - 0.5)).min(t - 0.05);
        let lift = if f < 0.02 || f > 0.98 { 0.1 } else { 0.0 };
        let (r3, r4) = if cabin { (Vec3::new(wb - s.inset, t - 0.05, z), Vec3::new((wb - s.inset) * 0.62, t, z)) } else { (Vec3::new(wb * 0.9, t - 0.01, z), Vec3::new(wb * 0.45, t + 0.015, z)) };
        [Vec3::new(wb * 0.84, s.clear + lift, z), Vec3::new(wb, s.clear + 0.15 + lift, z), Vec3::new(wb, ys, z), r3, r4]
    };
    let mir = |p: Vec3| Vec3::new(-p.x, p.y, p.z);
    for w in fs.windows(2) {
        let (a, b) = (ring(w[0]), ring(w[1]));
        let mid = (w[0] + w[1]) * 0.5;
        let cabin = mid > s.hood_end && mid < s.deck0;
        let screen = (mid > s.hood_end && mid < s.roof0) || (mid > s.roof1 && mid < s.deck0);
        let bed = s.bed && mid > s.deck0;
        for j in 0..4 {
            let col = match j {
                0 => trim,
                2 if cabin && mid < s.glass_to => glass,
                3 if screen && mid < s.glass_to.max(s.roof0) => glass,
                3 if bed => trim,
                _ => paint,
            };
            m.poly(&[a[j], b[j], b[j + 1], a[j + 1]], col);
            m.poly(&[mir(a[j + 1]), mir(b[j + 1]), mir(b[j]), mir(a[j])], col);
        }
        let col = if screen && mid < s.glass_to.max(s.roof0) { glass } else if bed { trim } else { paint };
        m.poly(&[a[4], b[4], mir(b[4]), mir(a[4])], col);
        m.poly(&[mir(a[0]), mir(b[0]), b[0], a[0]], trim);
    }
    // Nose and tail.
    for (f, dir) in [(0.0f32, -1.0f32), (1.0, 1.0)] {
        let r = ring(f);
        let mut loop_: Vec<Vec3> = r.to_vec();
        loop_.extend(r.iter().rev().map(|p| mir(*p)));
        if dir > 0.0 {
            loop_.reverse();
        }
        m.poly(&loop_, paint);
        let z = r[0].z + dir * 0.012;
        let (y0, y1) = (r[2].y - 0.2, r[2].y - 0.04);
        let q = |x0: f32, x1: f32, ya: f32, yb: f32, col: [f32; 4], m: &mut MB| {
            let p = [Vec3::new(x0, ya, z), Vec3::new(x1, ya, z), Vec3::new(x1, yb, z), Vec3::new(x0, yb, z)];
            m.poly(&if dir < 0.0 { [p[1], p[0], p[3], p[2]] } else { p }, col);
        };
        let wx = r[1].x;
        if dir < 0.0 {
            q(wx * 0.42, wx * 0.86, y0, y1, [0.9, 0.9, 0.82, 0.25], &mut m);
            q(-wx * 0.86, -wx * 0.42, y0, y1, [0.9, 0.9, 0.82, 0.25], &mut m);
            q(-wx * 0.36, wx * 0.36, s.clear + 0.22, y1 - 0.03, trim, &mut m);
            if s.flat_front {
                q(-wx * 0.9, wx * 0.9, s.belt + 0.08, s.h - 0.3, glass, &mut m);
            }
        } else {
            q(wx * 0.5, wx * 0.9, y0, y1, [0.45, 0.01, 0.01, 0.75], &mut m);
            q(-wx * 0.9, -wx * 0.5, y0, y1, [0.45, 0.01, 0.01, 0.75], &mut m);
            if s.flat_front {
                q(-wx * 0.8, wx * 0.8, s.belt + 0.1, s.h - 0.35, glass, &mut m);
            }
        }
        q(-wx * 0.96, wx * 0.96, s.clear + 0.1, s.clear + 0.3, trim, &mut m);
    }
    m.smooth(0.82);
    // Wheels, with dark arches on the body above them.
    for &af in s.axles {
        let z = (af - 0.5) * s.len;
        let wb = lerp_knots(&wide, af);
        for side in [-1.0f32, 1.0] {
            let (x0, x1) = (side * (wb - 0.2), side * (wb + 0.025));
            const N: usize = 12;
            let pt = |k: usize, x: f32, r: f32| {
                let a = k as f32 / N as f32 * std::f32::consts::TAU;
                Vec3::new(x, s.wheel_r + a.sin() * r, z + a.cos() * r)
            };
            for k in 0..N {
                let (a, b) = (k, (k + 1) % N);
                let quad = [pt(a, x0, s.wheel_r), pt(b, x0, s.wheel_r), pt(b, x1, s.wheel_r), pt(a, x1, s.wheel_r)];
                m.poly(&if side > 0.0 { quad } else { [quad[1], quad[0], quad[3], quad[2]] }, tyre);
                let hub = s.wheel_r * 0.62;
                let ring = [pt(a, x1, s.wheel_r), pt(b, x1, s.wheel_r), pt(b, x1, hub), pt(a, x1, hub)];
                m.poly(&if side > 0.0 { ring } else { [ring[1], ring[0], ring[3], ring[2]] }, tyre);
                let disc = [Vec3::new(x1 - side * 0.02, s.wheel_r, z), pt(a, x1, hub), pt(b, x1, hub)];
                m.poly(&if side > 0.0 { [disc[0], disc[1], disc[2]] } else { [disc[0], disc[2], disc[1]] }, [0.33, 0.34, 0.35, 1.0]);
            }
            let xa = side * (wb + 0.006);
            let arch = s.wheel_r * 1.24;
            for k in 0..N / 2 {
                let tri = [Vec3::new(xa, s.wheel_r, z), pt(k, xa, arch), pt(k + 1, xa, arch)];
                m.poly(&if side > 0.0 { tri } else { [tri[0], tri[2], tri[1]] }, [0.008, 0.008, 0.008, 1.0]);
            }
        }
    }
    // Door mirrors.
    if !s.flat_front {
        let z = (s.roof0 - 0.5) * s.len - 0.1;
        for side in [-1.0f32, 1.0] {
            let x = side * (s.w + 0.07);
            m.cuboid(Vec3::new(x - 0.07, s.belt + 0.02, z - 0.05), Vec3::new(x + 0.07, s.belt + 0.14, z + 0.05), trim);
        }
    }
    if s.cargo {
        let (z0, z1) = ((s.deck0 + 0.015 - 0.5) * s.len, 0.495 * s.len);
        m.cuboid(Vec3::new(-s.w - 0.05, s.deck - 0.05, z0), Vec3::new(s.w + 0.05, s.deck + 2.1, z1), [0.62, 0.62, 0.6, 1.0]);
    }
    (m.v, m.i)
}

pub const BASE_SPEC: Spec = Spec { len: 4.7, w: 0.91, h: 1.43, clear: 0.2, belt: 0.92, hood: 0.86, deck: 0.95, hood_end: 0.3, roof0: 0.41, roof1: 0.68, deck0: 0.82, wheel_r: 0.33, axles: &[0.185, 0.8], inset: 0.2, glass_to: 2.0, bed: false, flat_front: false, cargo: false };

pub const SPECS: [Spec; N_MODELS] = [
    Spec { ..BASE_SPEC },
    Spec { len: 4.0, w: 0.88, h: 1.47, belt: 0.94, hood: 0.88, deck: 1.0, hood_end: 0.26, roof0: 0.38, roof1: 0.8, deck0: 0.96, axles: &[0.2, 0.82], ..BASE_SPEC },
    Spec { w: 0.96, h: 1.74, clear: 0.27, belt: 1.08, hood: 1.02, deck: 1.14, hood_end: 0.27, roof0: 0.37, roof1: 0.9, deck0: 0.985, wheel_r: 0.38, inset: 0.16, ..BASE_SPEC },
    Spec { len: 5.4, w: 0.98, h: 1.8, clear: 0.3, belt: 1.12, hood: 1.08, deck: 1.16, hood_end: 0.27, roof0: 0.35, roof1: 0.55, deck0: 0.6, wheel_r: 0.4, axles: &[0.17, 0.78], inset: 0.14, bed: true, ..BASE_SPEC },
    Spec { len: 5.2, w: 0.98, h: 2.0, clear: 0.24, belt: 1.12, hood: 1.05, deck: 1.96, hood_end: 0.1, roof0: 0.2, roof1: 0.98, deck0: 0.99, wheel_r: 0.36, axles: &[0.17, 0.78], inset: 0.1, glass_to: 0.4, ..BASE_SPEC },
    Spec { len: 4.4, w: 0.96, h: 1.2, clear: 0.13, belt: 0.8, hood: 0.72, deck: 0.92, hood_end: 0.36, roof0: 0.48, roof1: 0.62, deck0: 0.9, wheel_r: 0.34, axles: &[0.19, 0.8], inset: 0.26, ..BASE_SPEC },
    Spec { len: 11.5, w: 1.26, h: 3.1, clear: 0.34, belt: 1.5, hood: 1.5, deck: 3.05, hood_end: 0.006, roof0: 0.02, roof1: 0.985, deck0: 0.995, wheel_r: 0.5, axles: &[0.16, 0.74, 0.8], inset: 0.06, flat_front: true, ..BASE_SPEC },
    Spec { len: 7.6, w: 1.12, h: 2.5, clear: 0.4, belt: 1.5, hood: 1.42, deck: 1.05, hood_end: 0.05, roof0: 0.1, roof1: 0.25, deck0: 0.27, wheel_r: 0.46, axles: &[0.12, 0.75, 0.82], inset: 0.08, cargo: true, ..BASE_SPEC },
];

pub fn build_models() -> Vec<(Vec<MeshVertex>, Vec<u32>)> {
    SPECS.iter().map(vehicle).collect()
}

/// A lumpy boulder for terrain debris.
pub fn rock(seed: u32) -> (Vec<MeshVertex>, Vec<u32>) {
    let (vs, fs) = crate::trees::ico(1);
    let pts: Vec<Vec3> = vs
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let q = (*d * 2.2).round() / 2.2;
            (*d * 0.55 + q * 0.45) * (0.75 + 0.5 * hash(k as u32, seed)) * Vec3::new(1.0, 0.7 + 0.2 * hash(seed, 9), 0.85)
        })
        .collect();
    let mut m = MB::default();
    for f in &fs {
        let sh = 0.7 + 0.5 * hash(f[0] + f[1] * 31, seed);
        m.poly(&[pts[f[0] as usize], pts[f[1] as usize], pts[f[2] as usize]], [sh, sh, sh, 0.0]);
    }
    m.smooth(0.75);
    (m.v, m.i)
}

pub fn cube() -> (Vec<MeshVertex>, Vec<u32>) {
    let mut m = MB::default();
    m.cuboid(Vec3::splat(-1.0), Vec3::splat(1.0), [1.0, 1.0, 1.0, 0.0]);
    (m.v, m.i)
}

pub const M_CUBE: u32 = N_MODELS as u32;
pub const M_ROCK: u32 = N_MODELS as u32 + 1;
/// broken concrete (destruct::slab_meshes), after the cube and three rocks
pub const M_SLAB: u32 = N_MODELS as u32 + 4;
/// reinforcing bar sticking out of broken concrete (destruct::rebar_meshes)
pub const M_REBAR: u32 = M_SLAB + crate::destruct::N_SLAB;
pub const N_REBAR: u32 = 2;

// ------------------------------------------------------------------ traffic

#[derive(Clone)]
pub struct Car {
    pub path: u32,
    pub s: f32,
    pub fwd: bool,
    pub lane: u8,
    pub v: f32,
    pub vmul: f32,
    pub model: u8,
    pub col: [f32; 3],
    /// 0..1; only cars ranked under the current traffic level are out
    pub rank: f32,
    /// smoothed heading and the fading offset left over from the last turn
    pub dir: Vec3,
    pub shift: Vec3,
    gap: f32,
    /// Under the 3D tiles: how far the mesh's street is above our road here,
    /// and the frame it was found on (World::photo_car_lights).
    pub street: std::cell::Cell<(f32, u32)>,
}

fn density(class: u8) -> f32 {
    match class {
        0 => 1.0 / 42.0,
        1 => 1.0 / 48.0,
        2 => 1.0 / 55.0,
        3 => 1.0 / 75.0,
        4 => 1.0 / 110.0,
        5 => 1.0 / 210.0,
        _ => 1.0 / 800.0,
    }
}

pub fn livery(model: u8, h: f32) -> [f32; 3] {
    match model {
        6 => [[0.75, 0.75, 0.72], [0.5, 0.03, 0.02], [0.62, 0.42, 0.02], [0.03, 0.12, 0.35], [0.6, 0.22, 0.02]][(h * 5.0) as usize % 5],
        7 => [[0.75, 0.75, 0.72], [0.4, 0.4, 0.4], [0.05, 0.1, 0.3], [0.4, 0.04, 0.03]][(h * 4.0) as usize % 4],
        _ => paint(h),
    }
}

fn place(net: &RoadNet, c: &mut Car, seed: u32) -> bool {
    // Pick a road with probability proportional to how much traffic it carries.
    let total: f32 = net.paths.iter().map(|p| p.len * density(p.class) * p.lanes as f32 * if p.oneway { 1.0 } else { 2.0 }).sum();
    if total <= 0.0 {
        return false;
    }
    let mut pick = hash(seed, 1) * total;
    for (i, p) in net.paths.iter().enumerate() {
        let w = p.len * density(p.class) * p.lanes as f32 * if p.oneway { 1.0 } else { 2.0 };
        if pick < w {
            c.path = i as u32;
            c.s = hash(seed, 2) * p.len;
            c.fwd = p.oneway || hash(seed, 3) < 0.5;
            c.lane = ((hash(seed, 4) * p.lanes as f32) as u8).min(p.lanes - 1);
            c.v = p.speed * c.vmul;
            c.shift = Vec3::ZERO;
            c.dir = Vec3::ZERO;
            return true;
        }
        pick -= w;
    }
    false
}

pub fn spawn(net: &RoadNet, tile_seed: u32) -> Vec<Car> {
    let want: f32 = net.paths.iter().map(|p| p.len * density(p.class) * p.lanes as f32 * if p.oneway { 1.0 } else { 2.0 }).sum();
    let n = (want as usize).min(520);
    let mut cars = Vec::with_capacity(n);
    for k in 0..n as u32 {
        let seed = tile_seed ^ k.wrapping_mul(2654435761);
        let mut c = Car { path: 0, s: 0.0, fwd: true, lane: 0, v: 0.0, vmul: 0.82 + 0.36 * hash(seed, 5), model: 0, col: [0.5; 3], rank: hash(seed, 6), dir: Vec3::ZERO, shift: Vec3::ZERO, gap: 1e9, street: std::cell::Cell::new((0.0, u32::MAX)) };
        if place(net, &mut c, seed) {
            c.model = car_model(hash(seed, 7), net.paths[c.path as usize].class <= 2);
            c.col = livery(c.model, hash(seed, 8));
            cars.push(c);
        }
    }
    cars
}

/// Where a car sits and which way it points, relative to the tile centre.
pub fn pose(net: &RoadNet, c: &Car, up: Vec3) -> (Vec3, Vec3) {
    let p = &net.paths[c.path as usize];
    let (pos, _) = p.sample(c.s);
    let (a, _) = p.sample(c.s - 2.4);
    let (b, _) = p.sample(c.s + 2.4);
    let mut d = (b - a).normalize_or_zero();
    if !c.fwd {
        d = -d;
    }
    let right = d.cross(up).normalize_or_zero();
    let off = if p.oneway { (c.lane as f32 + 0.5 - p.lanes as f32 * 0.5) * p.lane_w } else { (c.lane as f32 + 0.5) * p.lane_w * if net.left { -1.0 } else { 1.0 } };
    (pos + right * off, d)
}

/// Pick the way onward from a node. Returns (path, distance along, forward).
fn onward(net: &RoadNet, links: &[Link], from: u32, seed: u32) -> Option<(u32, f32, bool)> {
    let mut opts: Vec<(u32, f32, bool, f32)> = Vec::new();
    for l in links {
        if l.path == from {
            continue;
        }
        let p = &net.paths[l.path as usize];
        let w = if p.class >= 6 { 0.2 } else { 1.0 };
        match l.end {
            0 => opts.push((l.path, 0.0, true, w)),
            1 => {
                if !p.oneway {
                    opts.push((l.path, p.len, false, w))
                }
            }
            _ => {
                opts.push((l.path, l.at, true, w));
                if !p.oneway {
                    opts.push((l.path, l.at, false, w));
                }
            }
        }
    }
    let total: f32 = opts.iter().map(|o| o.3).sum();
    let mut pick = hash(seed, 11) * total;
    for o in &opts {
        if pick < o.3 {
            return Some((o.0, o.1, o.2));
        }
        pick -= o.3;
    }
    None
}

/// A car that drove off the edge of its tile: which edge, at which node.
pub struct Exit {
    pub car: Car,
    pub edge: u8,
    pub key: u64,
}

/// Put a car arriving from a neighbouring tile onto this tile's roads.
pub fn enter(net: &RoadNet, mut car: Car, key: u64, up: Vec3, was: Vec3) -> Option<Car> {
    for k in near_keys(key) {
        if let Some(links) = net.nodes.get(&k) {
            if let Some((p, s, fwd)) = onward(net, links, u32::MAX, (car.rank * 1e6) as u32 ^ k as u32) {
                car.path = p;
                car.s = s;
                car.fwd = fwd;
                car.lane = car.lane.min(net.paths[p as usize].lanes - 1);
                car.shift = was - pose(net, &car, up).0;
                if car.shift.length() > 12.0 {
                    car.shift = Vec3::ZERO;
                }
                return Some(car);
            }
        }
    }
    None
}

pub fn respawn(net: &RoadNet, car: &mut Car, seed: u32) {
    place(net, car, seed);
}

/// Advance every car of one tile. Cars that leave the tile are returned.
pub fn step(net: &RoadNet, cars: &mut Vec<Car>, dt: f32, up: Vec3, tick: u32) -> Vec<Exit> {
    // Following distance: sort the cars of each lane and look at the one in front.
    let mut order: Vec<usize> = (0..cars.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (&cars[a], &cars[b]);
        (x.path, x.fwd, x.lane).cmp(&(y.path, y.fwd, y.lane)).then(x.s.total_cmp(&y.s))
    });
    for c in cars.iter_mut() {
        c.gap = 1e9;
    }
    for w in order.windows(2) {
        let (a, b) = (w[0], w[1]);
        if (cars[a].path, cars[a].fwd, cars[a].lane) == (cars[b].path, cars[b].fwd, cars[b].lane) {
            let g = cars[b].s - cars[a].s - (CAR_LEN[cars[a].model as usize] + CAR_LEN[cars[b].model as usize]) * 0.5;
            if cars[a].fwd { cars[a].gap = g } else { cars[b].gap = g }
        }
    }
    let mut exits = Vec::new();
    let mut i = 0;
    while i < cars.len() {
        let seed = tick.wrapping_mul(31).wrapping_add(i as u32 * 7919);
        let c = &mut cars[i];
        let p = &net.paths[c.path as usize];
        let ahead = if c.fwd { p.len - c.s } else { c.s };
        let mut target = p.speed * c.vmul;
        // Ease off for the junction at the end of the road, and for the car in front.
        if ahead < 22.0 && p.class >= 2 {
            target = target.min(4.5 + ahead * 0.45);
        }
        target = target.min((c.gap - 2.5).max(0.0) * 0.9);
        c.v += (target - c.v).clamp(-7.0 * dt, 2.6 * dt);
        c.v = c.v.max(0.0);
        let before = c.s;
        c.s += c.v * dt * if c.fwd { 1.0 } else { -1.0 };
        c.shift *= (-dt * 2.2).exp();
        let was = pose(net, c, up).0 + c.shift;
        let mut next: Option<(u32, f32, bool)> = None;
        let mut gone = false;
        if c.s >= p.len || c.s <= 0.0 {
            let end = (c.s >= p.len) as usize;
            let key = p.nodes[end];
            next = net.nodes.get(&key).and_then(|l| onward(net, l, c.path, seed));
            if next.is_none() {
                if p.edge[end] != 0 {
                    gone = true;
                    exits.push(Exit { car: c.clone(), edge: p.edge[end], key });
                } else if !p.oneway {
                    // Dead end: turn round.
                    c.fwd = !c.fwd;
                    c.s = c.s.clamp(0.0, p.len);
                    c.v = 0.0;
                } else {
                    place(net, c, seed);
                }
            }
        } else if !p.junctions.is_empty() && hash(seed, 12) < 0.3 {
            // Passing a side street: sometimes take it.
            let (lo, hi) = (before.min(c.s), before.max(c.s));
            if let Some((_, key)) = p.junctions.iter().find(|j| j.0 > lo && j.0 <= hi) {
                next = net.nodes.get(key).and_then(|l| {
                    let ends: Vec<Link> = l.iter().filter(|x| x.end != 2).copied().collect();
                    onward(net, &ends, c.path, seed)
                });
            }
        }
        if let Some((np, ns, nf)) = next {
            c.path = np;
            c.s = ns;
            c.fwd = nf;
            c.lane = c.lane.min(net.paths[np as usize].lanes - 1);
            c.v = c.v.min(7.0);
            c.shift = was - pose(net, c, up).0;
        }
        if gone {
            cars.swap_remove(i);
        } else {
            i += 1;
        }
    }
    exits
}

/// How busy the roads are through the day (0..1) by local hour.
pub fn activity(hour: f32) -> f32 {
    let bump = |c: f32, w: f32| (-((hour - c) / w).powi(2)).exp();
    (0.1 + 0.9 * (bump(8.3, 2.2) + bump(17.5, 2.8) + 0.7 * bump(13.0, 3.0))).min(1.0)
}
