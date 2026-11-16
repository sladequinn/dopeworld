//! Multi-modal player controller: street-level walking/sprinting (`OnFoot`),
//! 4-wheel raycast suspension + RWD drift vehicle physics (`CarDrive`),
//! persistent parked vehicle hijacking, and aircraft boarding handoff (`PlaneFlight`).
use crate::airfield::{Light, L_WHITE};
use crate::audio::{Event, Params, Warn};
use crate::buildings::Hood;
use crate::cars::{self, CAR_LEN, N_MODELS};
use crate::geo;
use crate::render::Inst;
use crate::world::{StreetSample, World};
use glam::{DVec3, Vec2, Vec3};

pub const EYE_HEIGHT: f32 = 1.72;
pub const CAPSULE_RADIUS: f32 = 0.38;
pub const WALK_SPEED: f32 = 4.8;
pub const SPRINT_SPEED: f32 = 8.2;
pub const JUMP_IMPULSE: f32 = 4.8;
pub const GRAVITY: f32 = 9.81;
pub const STEP_HEIGHT: f32 = 0.45;
pub const HIJACK_RANGE: f64 = 3.5;
pub const PLANE_BOARD_RANGE: f64 = 9.5;
pub const LOW_EXIT_SPEED: f32 = 3.0;

/// Four wheel corner offsets in car body space `(right_x, forward_y)` in metres:
/// `[FL, FR, RL, RR]`.
pub const WHEEL_OFFSETS: [Vec2; 4] = [
    Vec2::new(-0.85, 1.45),
    Vec2::new(0.85, 1.45),
    Vec2::new(-0.85, -1.45),
    Vec2::new(0.85, -1.45),
];

pub const SUSP_REST_LEN: f32 = 0.45;
pub const SUSP_MAX_TRAVEL: f32 = 0.28;
pub const SUSP_K_SPRING: f32 = 34_000.0;
pub const SUSP_K_DAMP: f32 = 4_200.0;

/// 5-speed automatic transmission forward gear ratios.
pub const GEAR_RATIOS: [f32; 5] = [3.55, 2.18, 1.48, 1.12, 0.86];
pub const REVERSE_RATIO: f32 = -3.20;
pub const FINAL_DRIVE: f32 = 3.58;
pub const IDLE_RPM: f32 = 800.0;
pub const REDLINE_RPM: f32 = 6500.0;
pub const UPSHIFT_RPM: f32 = 6050.0;
pub const DOWNSHIFT_RPM: f32 = 2350.0;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum PlayMode {
    OnFoot,
    CarDrive,
    #[default]
    PlaneFlight,
}

/// Per-frame player control inputs for `OnFoot` and `CarDrive`.
#[derive(Copy, Clone, Default, Debug)]
pub struct PlayerInput {
    /// `W` (1.0) / `S` (-1.0) forward/back on foot, throttle/brake in car.
    pub move_fwd: f32,
    /// `D` (1.0) / `A` (-1.0) strafe right/left on foot, steer right/left in car.
    pub move_right: f32,
    /// `ShiftLeft`: sprint on foot, full-boost throttle in car.
    pub sprint: bool,
    /// `Space`: jump on foot.
    pub jump: bool,
    /// `B` or `Space` in car: rear-wheel handbrake lock for powerslides.
    pub handbrake: bool,
    /// `F` or `E` edge-triggered this frame: enter/exit vehicle or aircraft.
    pub interact: bool,
    /// Mouse-look or `Q`/`E` yaw delta (radians, positive = turn right / clockwise from above).
    pub yaw_delta: f32,
    /// Mouse-look pitch delta (radians, positive = look up).
    pub pitch_delta: f32,
}

/// Persistent state of the player's active or last-driven street car.
#[derive(Clone, Debug)]
pub struct ParkedCar {
    /// Planet-frame ECEF position (`DVec3`) on the road / terrain surface.
    pub pos: DVec3,
    /// Velocity in the local East-North-Up tangent frame (`x` East, `y` North, `z` Up) in m/s.
    pub vel: Vec3,
    /// Heading angle in the local East-North frame (`0` = North `+Y`, `PI/2` = East `+X`), radians.
    pub yaw: f32,
    /// Visual and physical body pitch angle (radians, positive = nose up, negative = nose dive).
    pub pitch: f32,
    /// Visual and physical body roll angle (radians, positive = roll right).
    pub roll: f32,
    /// Angular yaw rate (rad/s, positive = turning right).
    pub yaw_rate: f32,
    /// Pitch angular velocity (rad/s) for suspension spring-damper response.
    pub pitch_rate: f32,
    /// Roll angular velocity (rad/s) for suspension spring-damper response.
    pub roll_rate: f32,
    /// Current front wheel steering angle (radians, positive = right).
    pub steer: f32,
    /// Cumulative wheel rotation angle (radians).
    pub wheel_spin: f32,
    /// Compression (metres) of each of the 4 suspension struts `[FL, FR, RL, RR]`.
    pub wheel_comp: [f32; 4],
    /// Vehicle body mesh index (`0..cars::N_MODELS`).
    pub model: u8,
    /// Linear RGB body paint colour.
    pub col: [f32; 3],
    /// Current gear (`-1` = Reverse, `1..=5` = forward gears).
    pub gear: i8,
    /// Shift cooldown timer (seconds).
    pub shift_timer: f32,
    /// Current engine speed (`800..6500` RPM).
    pub rpm: f32,
    /// Smoothed throttle position (`0..1`).
    pub throttle: f32,
    /// Rear tire lateral slip ratio (`0..1`) for tire squeal / drift feedback.
    pub slip: f32,
    /// Continuous wall / curb scrape intensity (`0..1`).
    pub scrape: f32,
}

impl ParkedCar {
    pub fn new(pos: DVec3, yaw: f32, model: u8, col: [f32; 3]) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            yaw,
            pitch: 0.0,
            roll: 0.0,
            yaw_rate: 0.0,
            pitch_rate: 0.0,
            roll_rate: 0.0,
            steer: 0.0,
            wheel_spin: 0.0,
            wheel_comp: [SUSP_REST_LEN * 0.35; 4],
            model: model.min(N_MODELS as u8 - 1),
            col,
            gear: 1,
            shift_timer: 0.0,
            rpm: IDLE_RPM,
            throttle: 0.0,
            slip: 0.0,
            scrape: 0.0,
        }
    }

    /// Forward unit vector in the local East-North plane (`Vec2(east, north)`).
    pub fn forward_en(&self) -> Vec2 {
        let (s, c) = self.yaw.sin_cos();
        Vec2::new(s, c)
    }

    /// Right-hand unit vector in the local East-North plane (`Vec2(east, north)`).
    pub fn right_en(&self) -> Vec2 {
        let (s, c) = self.yaw.sin_cos();
        Vec2::new(c, -s)
    }

    /// Orthonormal `(right, up, forward)` basis in planet-frame `Vec3`, incorporating
    /// the car's local ENU heading (`yaw`) and suspension body tilt (`pitch`, `roll`).
    pub fn world_basis(&self) -> (Vec3, Vec3, Vec3) {
        let (east, north, up) = geo::enu(self.pos);
        let (e, n, u) = (east.as_vec3(), north.as_vec3(), up.as_vec3());
        let fwd_en = self.forward_en();
        let right_en = self.right_en();
        let fwd0 = (e * fwd_en.x + n * fwd_en.y).normalize_or_zero();
        let right0 = (e * right_en.x + n * right_en.y).normalize_or_zero();

        let (sp, cp) = self.pitch.sin_cos();
        let (sr, cr) = self.roll.sin_cos();

        let fwd = (fwd0 * cp + u * sp).normalize_or_zero();
        let right = (right0 * cr - u * sr).normalize_or_zero();
        let up_tilted = right.cross(fwd).normalize_or_zero();
        let right_ortho = fwd.cross(up_tilted).normalize_or_zero();
        (right_ortho, up_tilted, fwd)
    }
}

/// Multi-modal player state machine (`OnFoot`, `CarDrive`, `PlaneFlight`).
pub struct Player {
    pub mode: PlayMode,
    /// Player's feet (or active vehicle) planet-frame position (`DVec3`).
    pub pos: DVec3,
    /// Velocity in local East-North-Up tangent frame (`Vec3(east, north, up)`, m/s).
    pub vel: Vec3,
    /// Look / body yaw in local East-North frame (`0` = North `+Y`, `PI/2` = East `+X`), radians.
    pub yaw: f32,
    /// Look pitch angle in radians (`-1.4..1.4`, positive = look up).
    pub pitch: f32,
    /// Whether the on-foot capsule is supported by ground, bridge deck, or rooftop.
    pub grounded: bool,
    /// Head-bob phase (radians) while walking or sprinting on foot.
    pub walk_phase: f32,
    /// Persistent player car (either currently being driven in `CarDrive` or parked in the world).
    pub car: Option<ParkedCar>,
    /// Latest street sample at the player's position (elevation, bridge/rooftop flags, socio-economic `Hood`, `landuse`).
    pub last_sample: StreetSample,
    /// One-shot audio events emitted during the latest physics step.
    pub events: Vec<Event>,
}

impl Player {
    /// Create a player starting at `pos` in the given `PlayMode`.
    pub fn new(pos: DVec3, yaw: f32, mode: PlayMode) -> Self {
        Self {
            mode,
            pos,
            vel: Vec3::ZERO,
            yaw,
            pitch: 0.0,
            grounded: true,
            walk_phase: 0.0,
            car: None,
            last_sample: StreetSample::default(),
            events: Vec::new(),
        }
    }

    /// Spawn a persistent player car on the nearest road lane to `(lat, lon)` (radians),
    /// or directly at `(lat, lon)` if no road tile is loaded yet.
    pub fn spawn_car_near(&mut self, world: &World, lat: f64, lon: f64, model: u8, col: [f32; 3], enter_car: bool) {
        let (spawn_pos, yaw) = if let Some((road_pos, fwd_en)) = world.nearest_road_spawn(lat, lon) {
            let hdg = fwd_en.x.atan2(fwd_en.y);
            let sample = world.query_street(road_pos, geo::pos_to_lla(road_pos).2 as f32, 1.0);
            let (rlat, rlon, _) = geo::pos_to_lla(road_pos);
            (geo::lla_to_pos(rlat, rlon, sample.support_alt as f64), hdg)
        } else {
            let gh = world.ground(lat, lon).map(|g| g.h).unwrap_or(0.0);
            (geo::lla_to_pos(lat, lon, gh), self.yaw)
        };
        let car = ParkedCar::new(spawn_pos, yaw, model, col);
        if enter_car {
            self.pos = car.pos;
            self.yaw = car.yaw;
            self.vel = Vec3::ZERO;
            self.mode = PlayMode::CarDrive;
        }
        self.car = Some(car);
    }

    /// Eye / camera anchor in planet-frame ECEF (`DVec3`) and look forward/up vectors when on foot or driving.
    pub fn camera_pose(&self) -> (DVec3, DVec3, DVec3) {
        let (east, north, up) = geo::enu(self.pos);
        match self.mode {
            PlayMode::OnFoot => {
                let speed_h = self.vel.truncate().length();
                let bob = if self.grounded && speed_h > 0.2 {
                    (self.walk_phase.sin() * 0.035 * (speed_h / WALK_SPEED).min(1.6)) as f64
                } else {
                    0.0
                };
                let eye = self.pos + up * (EYE_HEIGHT as f64 + bob);
                let (sy, cy) = (self.yaw as f64).sin_cos();
                let (sp, cp) = (self.pitch as f64).sin_cos();
                let horiz = east * sy + north * cy;
                let fwd = (horiz * cp + up * sp).normalize();
                (eye, fwd, up)
            }
            PlayMode::CarDrive => {
                if let Some(car) = &self.car {
                    let (_, up_v, fwd_v) = car.world_basis();
                    let fwd = fwd_v.as_dvec3();
                    let up_d = up_v.as_dvec3();
                    let spec = &cars::SPECS[car.model as usize % N_MODELS];
                    // Chase camera behind and above the vehicle, following look yaw/pitch offset.
                    let chase_dist = (spec.len as f64 * 1.15).max(5.2);
                    let chase_up = (spec.h as f64 * 1.15).max(1.85);
                    let (sy, cy) = (self.yaw as f64).sin_cos();
                    let (sp, cp) = (self.pitch.clamp(-0.65, 0.45) as f64).sin_cos();
                    let look_horiz = (east * sy + north * cy).normalize();
                    let look_dir = (look_horiz * cp + up * sp).normalize();
                    let cam_pos = car.pos - look_horiz * chase_dist + up * chase_up;
                    let _ = (fwd, up_d);
                    (cam_pos, look_dir, up)
                } else {
                    (self.pos + up * EYE_HEIGHT as f64, north, up)
                }
            }
            PlayMode::PlaneFlight => (self.pos, north, up),
        }
    }

    /// Handle `KeyF` / `KeyE` vehicle & aircraft entry/exit transitions.
    ///
    /// `plane_pos` and `plane_ground_speed` describe the player's parked/active aircraft
    /// so the player can disembark when landed (`< 3.0 m/s`) and board it again from `OnFoot`.
    /// Returns `true` if a mode transition occurred.
    pub fn try_toggle_vehicle(
        &mut self,
        world: &mut World,
        plane_pos: Option<DVec3>,
        plane_fwd: Option<DVec3>,
        plane_ground_speed: f32,
    ) -> bool {
        match self.mode {
            PlayMode::OnFoot => {
                // 1. Check if within 3.5m of the player's own persistent parked car.
                let own_car_dist = self.car.as_ref().map(|c| (c.pos - self.pos).length()).unwrap_or(f64::MAX);
                if own_car_dist <= HIJACK_RANGE {
                    if let Some(car) = &mut self.car {
                        self.pos = car.pos;
                        self.yaw = car.yaw;
                        self.pitch = -0.12;
                        self.vel = car.vel;
                        self.mode = PlayMode::CarDrive;
                        self.events.push(Event::Clunk(0.6));
                        return true;
                    }
                }

                // 2. Check if within 3.5m of any ambient parked car in the loaded tiles.
                if let Some((p_pos, p_fwd, _p_up, model, col)) = world.take_nearest_parked_car(self.pos, HIJACK_RANGE) {
                    // Return the previous persistent car (if any) back to the world's parked list so it isn't lost.
                    if let Some(old) = self.car.take() {
                        let (_, old_up, old_fwd) = old.world_basis();
                        world.insert_parked_car(old.pos, old_fwd, old_up, old.model, old.col);
                    }
                    let (east, north, _) = geo::enu(p_pos);
                    let f_d = p_fwd.as_dvec3();
                    let yaw = (f_d.dot(east) as f32).atan2(f_d.dot(north) as f32);
                    let car = ParkedCar::new(p_pos, yaw, model, col);
                    self.pos = car.pos;
                    self.yaw = car.yaw;
                    self.pitch = -0.12;
                    self.vel = Vec3::ZERO;
                    self.car = Some(car);
                    self.mode = PlayMode::CarDrive;
                    self.events.push(Event::Clunk(0.75));
                    return true;
                }

                // 3. Check if within boarding range of the aircraft.
                if let Some(ppos) = plane_pos {
                    if (ppos - self.pos).length() <= PLANE_BOARD_RANGE {
                        self.pos = ppos;
                        self.vel = Vec3::ZERO;
                        self.mode = PlayMode::PlaneFlight;
                        self.events.push(Event::Clunk(0.8));
                        return true;
                    }
                }
                false
            }

            PlayMode::CarDrive => {
                let speed = self.car.as_ref().map(|c| c.vel.length()).unwrap_or(0.0);
                if speed >= LOW_EXIT_SPEED {
                    return false;
                }
                if let Some(car) = &mut self.car {
                    car.vel = Vec3::ZERO;
                    car.throttle = 0.0;
                    car.rpm = IDLE_RPM;
                    car.slip = 0.0;
                    car.scrape = 0.0;
                    let (east, north, _) = geo::enu(car.pos);
                    let right_en = car.right_en();
                    let left_world = -(east * right_en.x as f64 + north * right_en.y as f64).normalize();
                    // Exit 1.5m to the left of the driver's door.
                    let exit_pos = car.pos + left_world * 1.5;
                    let (_, _, alt) = geo::pos_to_lla(exit_pos);
                    let sample = world.query_street(exit_pos, alt as f32, CAPSULE_RADIUS);
                    let (elat, elon, _) = geo::pos_to_lla(exit_pos);
                    self.pos = geo::lla_to_pos(elat, elon, sample.support_alt as f64);
                    self.yaw = car.yaw;
                    self.pitch = 0.0;
                    self.vel = Vec3::ZERO;
                    self.grounded = true;
                    self.last_sample = sample;
                }
                self.mode = PlayMode::OnFoot;
                self.events.push(Event::Clunk(0.5));
                true
            }

            PlayMode::PlaneFlight => {
                if plane_ground_speed >= LOW_EXIT_SPEED {
                    return false;
                }
                let Some(ppos) = plane_pos else { return false };
                let (east, north, up) = geo::enu(ppos);
                let fwd = plane_fwd.unwrap_or(north);
                let right = fwd.cross(up).normalize_or_zero();
                let left = if right.length_squared() > 1e-6 { -right } else { -east };
                let exit_raw = ppos + left * 3.2;
                let (elat, elon, ealt) = geo::pos_to_lla(exit_raw);
                let sample = world.query_street(exit_raw, ealt as f32, CAPSULE_RADIUS);
                self.pos = geo::lla_to_pos(elat, elon, sample.support_alt as f64);
                self.yaw = (fwd.dot(east) as f32).atan2(fwd.dot(north) as f32);
                self.pitch = 0.0;
                self.vel = Vec3::ZERO;
                self.grounded = true;
                self.last_sample = sample;
                self.mode = PlayMode::OnFoot;
                self.events.push(Event::Clunk(0.6));
                true
            }
        }
    }

    /// Advance the active player controller (`OnFoot` or `CarDrive`) by `dt` seconds.
    pub fn step(
        &mut self,
        dt: f32,
        input: &PlayerInput,
        world: &mut World,
        plane_pos: Option<DVec3>,
        plane_fwd: Option<DVec3>,
        plane_ground_speed: f32,
    ) {
        self.events.clear();
        let dt = dt.clamp(1e-4, 0.1);

        if input.interact {
            self.try_toggle_vehicle(world, plane_pos, plane_fwd, plane_ground_speed);
        }

        match self.mode {
            PlayMode::OnFoot => self.step_on_foot(dt, input, world),
            PlayMode::CarDrive => self.step_car_drive(dt, input, world),
            PlayMode::PlaneFlight => {
                if let Some(ppos) = plane_pos {
                    self.pos = ppos;
                    let (_, _, alt) = geo::pos_to_lla(ppos);
                    self.last_sample = world.query_street(ppos, alt as f32, 1.0);
                }
            }
        }
    }

    /// `PlayMode::OnFoot`: capsule movement (`eye_height = 1.72m`), walk/sprint, jump,
    /// smooth curb/stair step-up (`<= 0.45m`), rooftop/bridge support, and 2D facade sliding.
    fn step_on_foot(&mut self, dt: f32, input: &PlayerInput, world: &World) {
        self.yaw = (self.yaw + input.yaw_delta).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + input.pitch_delta).clamp(-1.42, 1.42);

        let (sy, cy) = self.yaw.sin_cos();
        let fwd_en = Vec2::new(sy, cy);
        let right_en = Vec2::new(cy, -sy);

        let wish_dir = (fwd_en * input.move_fwd + right_en * input.move_right).clamp_length_max(1.0);
        let target_speed = if input.sprint { SPRINT_SPEED } else { WALK_SPEED };
        let target_vel_en = wish_dir * target_speed;

        let accel = if self.grounded { 28.0 } else { 6.0 };
        let cur_en = Vec2::new(self.vel.x, self.vel.y);
        let next_en = cur_en + (target_vel_en - cur_en) * (accel * dt).min(1.0);
        self.vel.x = next_en.x;
        self.vel.y = next_en.y;

        if self.grounded {
            if input.jump {
                self.vel.z = JUMP_IMPULSE;
                self.grounded = false;
            } else {
                self.vel.z = 0.0;
            }
        } else {
            self.vel.z = (self.vel.z - GRAVITY * dt).max(-55.0);
        }

        let (east, north, _) = geo::enu(self.pos);
        let (_, _, cur_alt_f64) = geo::pos_to_lla(self.pos);
        let cur_alt = cur_alt_f64 as f32;

        // Horizontal displacement on the planet sphere.
        let disp_world = east * (self.vel.x * dt) as f64 + north * (self.vel.y * dt) as f64;
        let next_alt_raw = cur_alt + self.vel.z * dt;
        let (nlat, nlon, _) = geo::pos_to_lla(self.pos + disp_world);
        let mut candidate_pos = geo::lla_to_pos(nlat, nlon, next_alt_raw as f64);

        // Query street surface and 2D building walls at the new position.
        // We pass `cur_alt + STEP_HEIGHT` as the query body altitude so steps/curbs up to 0.45m are Climbable.
        let probe_alt = if self.grounded { cur_alt + STEP_HEIGHT } else { next_alt_raw };
        let mut sample = world.query_street(candidate_pos, probe_alt, CAPSULE_RADIUS);

        // Resolve 2D building wall collision by pushing out along the East-North normal and sliding velocity.
        if let Some((normal_en, depth)) = sample.wall_push {
            let (ce, cn, _) = geo::enu(candidate_pos);
            let push_world = ce * (normal_en.x * depth) as f64 + cn * (normal_en.y * depth) as f64;
            let (plat, plon, _) = geo::pos_to_lla(candidate_pos + push_world);
            candidate_pos = geo::lla_to_pos(plat, plon, next_alt_raw as f64);
            let vn = self.vel.x * normal_en.x + self.vel.y * normal_en.y;
            if vn < 0.0 {
                self.vel.x -= normal_en.x * vn;
                self.vel.y -= normal_en.y * vn;
            }
            sample = world.query_street(candidate_pos, probe_alt, CAPSULE_RADIUS);
        }

        // Vertical support & smooth curb step-up (`<= 0.45m`).
        let support = sample.support_alt;
        let final_alt = if self.grounded && support >= next_alt_raw - 0.15 && support - cur_alt <= STEP_HEIGHT {
            // Smoothly step up curbs / stairs.
            self.vel.z = 0.0;
            cur_alt + (support - cur_alt) * (22.0 * dt).min(1.0)
        } else if next_alt_raw <= support {
            if self.vel.z < -3.5 {
                self.events.push(Event::Touch((-self.vel.z).min(12.0)));
            }
            self.grounded = true;
            self.vel.z = 0.0;
            support
        } else {
            self.grounded = false;
            next_alt_raw
        };

        let (flat, flon, _) = geo::pos_to_lla(candidate_pos);
        self.pos = geo::lla_to_pos(flat, flon, final_alt as f64);
        self.last_sample = sample;

        let horiz_speed = Vec2::new(self.vel.x, self.vel.y).length();
        if self.grounded && horiz_speed > 0.15 {
            self.walk_phase = (self.walk_phase + horiz_speed * dt * 2.4).rem_euclid(std::f32::consts::TAU);
        }
    }

    /// `PlayMode::CarDrive`: 4-wheel raycast suspension, 5-speed automatic transmission,
    /// speed-sensitive steering (`32 deg` -> `9 deg`), RWD powerslides, and building facade collisions.
    fn step_car_drive(&mut self, dt: f32, input: &PlayerInput, world: &World) {
        let Some(mut car) = self.car.take() else {
            self.mode = PlayMode::OnFoot;
            return;
        };

        let (east, north, _) = geo::enu(car.pos);
        let (_, _, cur_alt_f64) = geo::pos_to_lla(car.pos);
        let cur_alt = cur_alt_f64 as f32;

        let fwd_en = car.forward_en();
        let right_en = car.right_en();
        let vel_en = Vec2::new(car.vel.x, car.vel.y);
        let mut v_fwd = vel_en.dot(fwd_en);
        let mut v_lat = vel_en.dot(right_en);
        let speed = vel_en.length();

        // 1. Speed-sensitive front-wheel steering: 32 deg at low speed, tapering to 9 deg at highway speed (38 m/s).
        let speed_t = (speed / 38.0).clamp(0.0, 1.0);
        let max_steer = 32.0f32.to_radians() + (9.0f32.to_radians() - 32.0f32.to_radians()) * speed_t;
        let target_steer = input.move_right.clamp(-1.0, 1.0) * max_steer;
        let steer_rate = 7.5;
        car.steer += (target_steer - car.steer) * (steer_rate * dt).min(1.0);

        // 2. 5-Speed Automatic Gearbox & Drivetrain.
        car.shift_timer = (car.shift_timer - dt).max(0.0);
        let want_fwd = input.move_fwd.max(0.0);
        let want_rev = (-input.move_fwd).max(0.0);

        if car.gear == -1 {
            if want_fwd > 0.1 && v_fwd > -1.2 {
                car.gear = 1;
                car.shift_timer = 0.22;
            }
        } else if want_rev > 0.1 && v_fwd < 0.8 {
            car.gear = -1;
            car.shift_timer = 0.22;
        }

        let (throttle_in, brake_in) = if car.gear == -1 {
            (want_rev, want_fwd)
        } else {
            let boost = if input.sprint && want_fwd > 0.05 { 1.0 } else { want_fwd };
            (boost, want_rev)
        };

        let prev_throttle = car.throttle;
        car.throttle += (throttle_in - car.throttle) * (10.0 * dt).min(1.0);
        // Throttle-chop exhaust pop when lifting off from high RPM.
        if prev_throttle > 0.65 && throttle_in < 0.1 && car.rpm > 4200.0 {
            self.events.push(Event::Bump(0.18));
        }

        let spec = &cars::SPECS[car.model as usize % N_MODELS];
        let wheel_r = spec.wheel_r.max(0.28);
        let ratio = if car.gear == -1 {
            REVERSE_RATIO.abs()
        } else {
            GEAR_RATIOS[(car.gear.clamp(1, 5) - 1) as usize]
        };

        let wheel_rpm = (v_fwd.abs() / (std::f32::consts::TAU * wheel_r)) * 60.0;
        let coupled_rpm = wheel_rpm * ratio * FINAL_DRIVE;
        let free_rev = IDLE_RPM + car.throttle * (REDLINE_RPM - IDLE_RPM) * 0.45;
        let target_rpm = coupled_rpm.max(free_rev).clamp(IDLE_RPM, REDLINE_RPM);
        car.rpm += (target_rpm - car.rpm) * (12.0 * dt).min(1.0);

        // Automatic upshift / downshift with RPM drop.
        if car.gear >= 1 && car.shift_timer <= 0.0 {
            if car.gear < 5 && car.rpm >= UPSHIFT_RPM && car.throttle > 0.25 {
                let old_r = GEAR_RATIOS[(car.gear - 1) as usize];
                car.gear += 1;
                let new_r = GEAR_RATIOS[(car.gear - 1) as usize];
                car.rpm = (car.rpm * (new_r / old_r)).max(IDLE_RPM);
                car.shift_timer = 0.28;
            } else if car.gear > 1 && car.rpm <= DOWNSHIFT_RPM && v_fwd > 2.0 {
                let old_r = GEAR_RATIOS[(car.gear - 1) as usize];
                car.gear -= 1;
                let new_r = GEAR_RATIOS[(car.gear - 1) as usize];
                car.rpm = (car.rpm * (new_r / old_r)).min(REDLINE_RPM);
                car.shift_timer = 0.22;
            }
        }

        // Torque curve peaking around 4400 RPM.
        let rpm_norm = ((car.rpm - IDLE_RPM) / (REDLINE_RPM - IDLE_RPM)).clamp(0.0, 1.0);
        let torque_factor = 0.65 + 0.35 * (1.0 - ((rpm_norm - 0.62) / 0.5).powi(2)).max(0.0);
        let shift_cut = if car.shift_timer > 0.12 { 0.25 } else { 1.0 };
        let base_accel = if car.model == 5 { 11.5 } else if car.model >= 6 { 5.2 } else { 8.4 };
        let gear_mul = (ratio / GEAR_RATIOS[1]).clamp(0.48, 1.65);

        let drive_dir = if car.gear == -1 { -1.0 } else { 1.0 };
        let max_speed = if car.gear == -1 { 14.0 } else if car.model == 5 { 62.0 } else { 48.0 };
        let mut long_accel = if v_fwd * drive_dir < max_speed {
            drive_dir * car.throttle * base_accel * torque_factor * gear_mul * shift_cut
        } else {
            0.0
        };

        // Service brakes + aerodynamic drag + rolling resistance.
        if brake_in > 0.01 {
            let brake_decel = 18.5 * brake_in;
            if v_fwd.abs() > 0.15 {
                long_accel -= v_fwd.signum() * brake_decel;
            } else {
                v_fwd = 0.0;
            }
        }
        if input.handbrake {
            if v_fwd.abs() > 0.15 {
                long_accel -= v_fwd.signum() * 9.5;
            } else {
                v_fwd = 0.0;
            }
        }
        let drag = 0.0042 * v_fwd * v_fwd.abs() + 0.18 * v_fwd;
        long_accel -= drag;
        v_fwd += long_accel * dt;

        // 3. Tire Grip & RWD Powerslides.
        let wheelbase = 2.90f32;
        let heavy_power_oversteer = car.throttle > 0.72 && car.steer.abs() > 0.16 && v_fwd > 4.0;
        let rear_grip = if input.handbrake {
            2.4
        } else if heavy_power_oversteer {
            5.2
        } else {
            16.5
        };

        // Ackermann kinematic yaw rate blended with dynamic slip / powerslide yaw torque.
        let ackermann_yaw_rate = (v_fwd / wheelbase) * car.steer.tan();
        let oversteer_boost = if input.handbrake && speed > 3.5 {
            car.steer.signum() * (car.steer.abs() * 1.45) * (speed / 14.0).clamp(0.0, 1.3)
        } else if heavy_power_oversteer {
            car.steer * 0.65 * car.throttle
        } else {
            0.0
        };
        let target_yaw_rate = ackermann_yaw_rate + oversteer_boost;
        let yaw_stiffness = if input.handbrake { 6.5 } else { 11.0 };
        car.yaw_rate += (target_yaw_rate - car.yaw_rate) * (yaw_stiffness * dt).min(1.0);
        car.yaw = (car.yaw + car.yaw_rate * dt).rem_euclid(std::f32::consts::TAU);

        // Centripetal transfer from turning into lateral velocity + tire lateral restoring force.
        v_lat += v_fwd * car.yaw_rate * dt * if input.handbrake { 0.75 } else { 0.32 };
        let lat_correction = (-v_lat).clamp(-rear_grip * dt, rear_grip * dt);
        v_lat += lat_correction;
        car.slip = ((v_lat.abs() - 1.2) / 7.5).clamp(0.0, 1.0);
        if input.handbrake && speed > 4.0 {
            car.slip = car.slip.max(0.65);
        }

        let new_fwd_en = car.forward_en();
        let new_right_en = car.right_en();
        let new_vel_en = new_fwd_en * v_fwd + new_right_en * v_lat;
        car.vel.x = new_vel_en.x;
        car.vel.y = new_vel_en.y;
        car.wheel_spin = (car.wheel_spin + (v_fwd / wheel_r) * dt).rem_euclid(std::f32::consts::TAU);

        // 4. Advance horizontal position & sample 4-Wheel Raycast Suspension.
        let disp_world = east * (car.vel.x * dt) as f64 + north * (car.vel.y * dt) as f64;
        let next_pos_raw = car.pos + disp_world;
        let (nlat, nlon, _) = geo::pos_to_lla(next_pos_raw);
        let mut next_pos = geo::lla_to_pos(nlat, nlon, (cur_alt + car.vel.z * dt) as f64);

        let (ce, cn, _) = geo::enu(next_pos);
        let mut wheel_support = [0.0f32; 4];
        for (wi, off) in WHEEL_OFFSETS.iter().enumerate() {
            let w_en = new_right_en * off.x + new_fwd_en * off.y;
            let w_pos = next_pos + ce * w_en.x as f64 + cn * w_en.y as f64;
            let ws = world.query_street(w_pos, cur_alt + STEP_HEIGHT, 0.0);
            wheel_support[wi] = ws.support_alt;

            let wheel_world_z = cur_alt + off.y * car.pitch.sin() - off.x * car.roll.sin();
            let comp = (SUSP_REST_LEN - (wheel_world_z - ws.support_alt)).clamp(0.0, SUSP_REST_LEN + SUSP_MAX_TRAVEL);
            car.wheel_comp[wi] = comp;
        }

        let avg_support = (wheel_support[0] + wheel_support[1] + wheel_support[2] + wheel_support[3]) * 0.25;
        let front_support = (wheel_support[0] + wheel_support[1]) * 0.5;
        let rear_support = (wheel_support[2] + wheel_support[3]) * 0.5;
        let left_support = (wheel_support[0] + wheel_support[2]) * 0.5;
        let right_support = (wheel_support[1] + wheel_support[3]) * 0.5;

        // Terrain slope pitch/roll + dynamic squat under acceleration / nose-dive under braking + body roll in turns.
        let slope_pitch = ((front_support - rear_support) / 2.90).atan();
        let slope_roll = ((left_support - right_support) / 1.70).atan();
        let accel_squat = (long_accel * 0.0055).clamp(-0.085, 0.065);
        let lat_accel = v_fwd * car.yaw_rate;
        let turn_roll = (-lat_accel * 0.0065).clamp(-0.11, 0.11);

        let target_pitch = (slope_pitch + accel_squat).clamp(-0.45, 0.45);
        let target_roll = (slope_roll + turn_roll).clamp(-0.45, 0.45);

        let k_ang = 48.0;
        let d_ang = 11.0;
        car.pitch_rate += ((target_pitch - car.pitch) * k_ang - car.pitch_rate * d_ang) * dt;
        car.roll_rate += ((target_roll - car.roll) * k_ang - car.roll_rate * d_ang) * dt;
        car.pitch += car.pitch_rate * dt;
        car.roll += car.roll_rate * dt;

        // Vertical suspension / gravity support.
        let mut next_alt = cur_alt + car.vel.z * dt;
        if next_alt <= avg_support + 0.08 {
            if car.vel.z < -3.0 {
                self.events.push(Event::Touch((-car.vel.z).min(15.0)));
            }
            next_alt = cur_alt + (avg_support - cur_alt) * (24.0 * dt).min(1.0);
            car.vel.z = 0.0;
            self.grounded = true;
        } else {
            car.vel.z = (car.vel.z - GRAVITY * dt).max(-55.0);
            self.grounded = false;
        }

        // 5. Building Wall & Curb Collisions.
        let car_radius = (spec.w * 1.15).max(1.05);
        let mut center_sample = world.query_street(next_pos, next_alt + STEP_HEIGHT, car_radius);
        car.scrape *= (-dt * 6.0).exp();

        if let Some((normal_en, depth)) = center_sample.wall_push {
            let push_world = ce * (normal_en.x * depth) as f64 + cn * (normal_en.y * depth) as f64;
            next_pos += push_world;

            let vel_cur = Vec2::new(car.vel.x, car.vel.y);
            let vn = vel_cur.dot(normal_en);
            if vn < 0.0 {
                let impact = -vn;
                // Reflect with low restitution (0.18) and slide along the facade tangent with friction.
                let tangent_en = Vec2::new(-normal_en.y, normal_en.x);
                let vt = vel_cur.dot(tangent_en) * 0.86;
                let bounced = normal_en * (-vn * 0.18) + tangent_en * vt;
                car.vel.x = bounced.x;
                car.vel.y = bounced.y;

                // Apply angular yaw impulse when a front corner clips a building wall.
                let wall_dir = -normal_en;
                let corner_lever = new_fwd_en.perp_dot(wall_dir);
                car.yaw_rate += corner_lever * (impact * 0.22).clamp(-2.8, 2.8);

                if impact > 2.2 {
                    self.events.push(Event::Bump((impact * 0.35).clamp(0.25, 3.5)));
                }
                car.scrape = (car.vel.truncate().length() / 12.0).clamp(0.15, 1.0);
            }
            center_sample = world.query_street(next_pos, next_alt + STEP_HEIGHT, car_radius);
        }

        let (flat, flon, _) = geo::pos_to_lla(next_pos);
        car.pos = geo::lla_to_pos(flat, flon, next_alt as f64);

        self.pos = car.pos;
        self.vel = car.vel;
        self.yaw = car.yaw;
        self.last_sample = center_sample;
        self.car = Some(car);
    }

    /// Emit the player's persistent car (whether actively driven in `CarDrive` or parked while
    /// the player walks around it in `OnFoot`) into the instanced solid pipeline (`out`) and
    /// dynamic headlight/taillight list (`lights`).
    pub fn emit_car_instance(&self, cam: DVec3, sun_elev: f64, out: &mut Vec<(u32, Inst)>, lights: &mut Vec<Light>) {
        let Some(car) = &self.car else { return };
        let o = (car.pos - cam).as_vec3();
        if o.length_squared() > 2500.0 * 2500.0 {
            return;
        }
        let (right, up, fwd) = car.world_basis();
        out.push((
            car.model as u32,
            Inst {
                m: [
                    [right.x, up.x, -fwd.x, o.x],
                    [right.y, up.y, -fwd.y, o.y],
                    [right.z, up.z, -fwd.z, o.z],
                ],
                tint: [car.col[0], car.col[1], car.col[2], 0.0],
                ext: [0.0, 0.0, 0.0, 1.0],
            },
        ));

        // Emit headlights and brake/tail lights when driving or at dusk/night.
        let active_drive = self.mode == PlayMode::CarDrive;
        if active_drive || sun_elev < 0.18 {
            let len = CAR_LEN[car.model as usize % N_MODELS];
            let half_w = if len > 7.0 { 1.0 } else { 0.62 };
            let head_inten = if active_drive { 85.0 } else { 40.0 };
            let tail_inten = if active_drive { 22.0 } else { 10.0 };
            for side in [-half_w, half_w] {
                lights.push(Light {
                    pos: (o + fwd * (len * 0.5 - 0.08) + right * side + up * 0.68).into(),
                    inten: head_inten,
                    dir: (fwd - up * 0.14).normalize_or_zero().into(),
                    spread: -0.42,
                    col: [L_WHITE[0], L_WHITE[1], L_WHITE[2], 0],
                    fx: [0, 0, 255, 0],
                });
                lights.push(Light {
                    pos: (o - fwd * (len * 0.5 - 0.08) + right * side + up * 0.82).into(),
                    inten: tail_inten,
                    dir: (-fwd - up * 0.12).normalize_or_zero().into(),
                    spread: -0.35,
                    col: [255, 24, 14, 0],
                    fx: [0, 0, 255, 0],
                });
            }
        }
    }

    /// Synthesise `audio::Params` for `CarDrive` or `OnFoot` so `src/audio.rs` produces
    /// realistic engine RPM, exhaust note, tire rolling/braking squeal, and wall scrape audio.
    pub fn audio_params(&self, master: f32, cam_dist: f32) -> Params {
        match self.mode {
            PlayMode::CarDrive => {
                if let Some(car) = &self.car {
                    let speed = car.vel.length();
                    let power = ((car.rpm - IDLE_RPM) / (REDLINE_RPM - IDLE_RPM)).clamp(0.0, 1.0) * (0.25 + 0.75 * car.throttle);
                    Params {
                        live: 1.0,
                        master,
                        rpm: car.rpm * 0.48,
                        power,
                        throttle: car.throttle,
                        health: 1.0,
                        prop: 0.0,
                        airspeed: speed,
                        vne: 95.0,
                        jet: 0.0,
                        n1: 0.0,
                        fighter: 0.0,
                        burner: 0.0,
                        g_load: 1.0,
                        rough: (speed / 28.0).clamp(0.0, 0.35),
                        jolt: 0.0,
                        buffet: 0.0,
                        flutter: 0.0,
                        gear: 1.0,
                        gear_moving: 0.0,
                        flaps: 0.0,
                        flaps_moving: 0.0,
                        rolling: if self.grounded { speed } else { 0.0 },
                        braking: car.slip,
                        scrape: car.scrape,
                        foliage: 0.0,
                        cracked: 0.0,
                        cabin: 0.45,
                        dist: cam_dist,
                        fire: 0.0,
                        warn: Warn::default(),
                    }
                } else {
                    Params { live: 0.0, master, ..Default::default() }
                }
            }
            PlayMode::OnFoot => Params {
                live: 0.0,
                master,
                airspeed: self.vel.length(),
                dist: cam_dist,
                ..Default::default()
            },
            PlayMode::PlaneFlight => Params { live: 0.0, master, ..Default::default() },
        }
    }

    /// Current socio-economic neighbourhood metrics and landuse byte under the player.
    pub fn neighborhood(&self) -> (Hood, u8) {
        (self.last_sample.hood, self.last_sample.landuse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::{Loader, Sources, DATA_ZOOM};
    use std::sync::Arc;

    fn test_world() -> World {
        let dir = std::env::temp_dir().join("fablegame-player-test-cache");
        let src = Arc::new(Sources::new(crate::fetch::Fetcher::new(dir, true), String::new()));
        let mut w = World::new(Loader::new(src, 0), DATA_ZOOM);
        w.flat = Some(15.0);
        w
    }

    #[test]
    fn on_foot_walking_sprinting_and_jumping() {
        let mut world = test_world();
        let start = geo::lla_to_pos(0.6, -1.2, 15.0);
        let mut player = Player::new(start, 0.0, PlayMode::OnFoot);

        // Sprint north for 1 second.
        let input = PlayerInput { move_fwd: 1.0, sprint: true, ..Default::default() };
        for _ in 0..60 {
            player.step(1.0 / 60.0, &input, &mut world, None, None, 0.0);
        }
        let (_, north, _) = geo::enu(start);
        let moved_north = (player.pos - start).dot(north);
        assert!(moved_north > 6.5, "expected sprint distance > 6.5m, got {moved_north}");
        assert!(player.grounded);

        // Jump impulse.
        let jump_in = PlayerInput { jump: true, ..Default::default() };
        player.step(1.0 / 60.0, &jump_in, &mut world, None, None, 0.0);
        assert!(!player.grounded && player.vel.z > 4.0);
    }

    #[test]
    fn car_drive_gearbox_drift_and_enter_exit() {
        let mut world = test_world();
        let lat = 0.6;
        let lon = -1.2;
        let start = geo::lla_to_pos(lat, lon, 15.0);
        let mut player = Player::new(start, 0.0, PlayMode::OnFoot);
        player.spawn_car_near(&world, lat, lon, 0, [0.8, 0.1, 0.1], true);
        assert_eq!(player.mode, PlayMode::CarDrive);

        // Accelerate hard for 4 seconds: verify speed builds, pitch squats, and gearbox upshifts.
        let accel_in = PlayerInput { move_fwd: 1.0, sprint: true, ..Default::default() };
        for _ in 0..240 {
            player.step(1.0 / 60.0, &accel_in, &mut world, None, None, 0.0);
        }
        let car = player.car.as_ref().unwrap();
        assert!(car.vel.length() > 18.0, "car should accelerate: vel={}", car.vel.length());
        assert!(car.gear >= 2, "5-speed gearbox should upshift: gear={}", car.gear);
        assert!(car.rpm >= IDLE_RPM && car.rpm <= REDLINE_RPM);

        // Apply handbrake + right steering: verify powerslide slip and yaw rate.
        let drift_in = PlayerInput { move_fwd: 0.0, move_right: 1.0, handbrake: true, ..Default::default() };
        for _ in 0..20 {
            player.step(1.0 / 60.0, &drift_in, &mut world, None, None, 0.0);
        }
        let car = player.car.as_ref().unwrap();
        assert!(car.slip > 0.3 && car.yaw_rate > 0.1);

        // Brake to a stop and press F (`interact: true`) to exit 1.5m left of the driver's door.
        player.car.as_mut().unwrap().vel = Vec3::ZERO;
        let car_pos = player.car.as_ref().unwrap().pos;
        let exit_in = PlayerInput { interact: true, ..Default::default() };
        player.step(1.0 / 60.0, &exit_in, &mut world, None, None, 0.0);
        assert_eq!(player.mode, PlayMode::OnFoot);
        let exit_dist = (player.pos - car_pos).length();
        assert!((exit_dist - 1.5).abs() < 0.25, "expected exit 1.5m from car, got {exit_dist}");

        // Press F again within 3.5m to re-enter the parked car.
        player.step(1.0 / 60.0, &exit_in, &mut world, None, None, 0.0);
        assert_eq!(player.mode, PlayMode::CarDrive);
    }
}
