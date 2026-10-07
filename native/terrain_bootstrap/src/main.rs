//! CPU-only export of the pinned World Builder physical atlas.
//! No viewer, fine landforms, camera-dependent fields, or neural inference.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, fs, io::{BufWriter, Write}, path::PathBuf};
use world_core::{geometry, World, WorldConfig, GENERATOR_VERSION};

#[derive(Deserialize)]
struct Request { seed: u64, style: String, resolution: u32, width: usize, height: usize }

#[derive(Serialize)]
struct Topology { land_fraction: f64, components: usize, largest_land_fraction: f64, significant_components: usize }

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn neighbours8(i: usize, n: u32) -> [usize; 8] {
    let size = n as usize*n as usize;
    let face = (i/size) as u8;
    let x = (i%size%n as usize) as i32;
    let y = (i%size/n as usize) as i32;
    std::array::from_fn(|k| {
        let (dx,dy)=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)][k];
        let (nx,ny)=(x+dx,y+dy);
        if nx>=0 && ny>=0 && nx<n as i32 && ny<n as i32 {
            geometry::cell_index(face,nx as u32,ny as u32,n)
        } else {
            geometry::direction_index(geometry::direction(face,2.*(nx as f64+0.5)/n as f64-1.,2.*(ny as f64+0.5)/n as f64-1.),n)
        }
    })
}

fn topology(world: &World) -> Topology {
    let mut seen=vec![false;world.cells.len()];
    let mut components=Vec::new();
    for i in 0..world.cells.len() {
        if seen[i] || world.cells[i].height_m<=0. { continue; }
        seen[i]=true;
        let mut queue=VecDeque::from([i]);
        let mut area=0.;
        while let Some(j)=queue.pop_front() {
            area+=world.cells[j].area_m2;
            for next in neighbours8(j,world.config.resolution) {
                if !seen[next] && world.cells[next].height_m>0. {
                    seen[next]=true; queue.push_back(next);
                }
            }
        }
        components.push(area);
    }
    let total: f64=components.iter().sum();
    Topology {
        land_fraction:total/world.stats.surface_area_m2,
        components:components.len(),
        largest_land_fraction:components.iter().copied().fold(0.,f64::max)/total.max(1.),
        significant_components:components.iter().filter(|&&a|a>0.05*total).count(),
    }
}

fn acceptable(style: &str,t: &Topology)->bool {
    match style {
        "gondwana" => t.largest_land_fraction>=0.85,
        "continents" => t.largest_land_fraction<=0.65 && t.significant_components>=3,
        "earthlike" => t.largest_land_fraction<=0.75 && t.significant_components>=2,
        "archipelago" => t.largest_land_fraction<=0.15 && t.components>=100,
        _ => false,
    }
}

fn config(request: &Request,seed:u64)->Result<WorldConfig> {
    let (plates,scale,fragmentation,ocean)=match request.style.as_str() {
        "gondwana" => (3,0.35,0.12,0.65),
        "continents" => (14,0.85,0.85,0.65),
        "earthlike" => (14,1.0,0.8,0.71),
        "archipelago" => (28,3.0,1.6,0.82),
        _ => bail!("unsupported style"),
    };
    Ok(WorldConfig {seed,resolution:request.resolution,radius_m:6_371_000.,
        gravity_m_s2:9.81,plate_count:plates,continent_scale:scale,continent_fragmentation:fragmentation,
        ocean_fraction:ocean,erosion_steps:0,detail_m:0.,coast_detail_ratio:0.,
        archipelago_density:0.,fjord_strength:0.,regional_lake_density:0.,
        ..Default::default()})
}

// Bilinear reconstruction between physical cell centres. Virtual samples beyond
// a cube face are projected to the adjacent native face, rather than clamped.
fn face_height(world: &World,p:[f64;3],face:u8)->f64 {
    let n=world.config.resolution;
    let [normal,axis_u,axis_v]=geometry::FACE_BASES[face as usize];
    let divisor=geometry::dot(p,normal);
    let u=geometry::dot(p,axis_u)/divisor;
    let v=geometry::dot(p,axis_v)/divisor;
    let x=(u+1.)*0.5*n as f64-0.5;
    let y=(v+1.)*0.5*n as f64-0.5;
    let ix=x.floor() as i32; let iy=y.floor() as i32;
    let tx=x-ix as f64; let ty=y-iy as f64;
    let at=|nx:i32,ny:i32| {
        let index=if nx>=0 && ny>=0 && nx<n as i32 && ny<n as i32 {
            geometry::cell_index(face,nx as u32,ny as u32,n)
        } else {
            geometry::direction_index(geometry::direction(face,2.*(nx as f64+0.5)/n as f64-1.,2.*(ny as f64+0.5)/n as f64-1.),n)
        };
        world.cells[index].height_m as f64
    };
    (1.-ty)*((1.-tx)*at(ix,iy)+tx*at(ix+1,iy))+ty*((1.-tx)*at(ix,iy+1)+tx*at(ix+1,iy+1))
}

// Overlap a one-cell strip of adjacent reconstructions. A hard dominant-face
// switch between nearest ghost samples produces small discontinuities even
// when each face is bilinear. Partition weights make edges and corners agree.
fn coarse_height(world: &World,p:[f64;3])->f32 {
    let dominant=p.iter().map(|value|value.abs()).fold(0.,f64::max);
    let strip=dominant/world.config.resolution as f64;
    let mut total=0.;
    let mut weight_sum=0.;
    for face in 0..6u8 {
        let score=geometry::dot(p,geometry::FACE_BASES[face as usize][0]);
        let weight=((score-(dominant-strip))/strip).clamp(0.,1.);
        if weight>0. {
            total+=weight*face_height(world,p,face);
            weight_sum+=weight;
        }
    }
    (total/weight_sum) as f32
}

fn main()->Result<()> {
    let arguments:Vec<_>=std::env::args().skip(1).collect();
    if arguments.len()!=2 { bail!("usage: terrain-bootstrap request.json output-directory"); }
    let request:Request=serde_json::from_slice(&fs::read(&arguments[0])?)?;
    if !(128..=512).contains(&request.resolution) || request.width<16 || request.width>4096 || request.height<8 || request.height>2048 { bail!("invalid export grid"); }
    let started=std::time::Instant::now();
    let mut selected=None; let mut attempts=Vec::new();
    for attempt in 0..12u64 {
        let seed=if attempt==0 {request.seed} else {mix(request.seed^attempt.wrapping_mul(0xd1b54a32d192ed03))};
        let world=World::generate(config(&request,seed)?)?;
        let topo=topology(&world);
        let passed=acceptable(&request.style,&topo);
        attempts.push(serde_json::json!({"attempt":attempt,"seed_u64":seed.to_string(),"topology":topo,"accepted":passed}));
        if passed {selected=Some((world,topo,attempt));break;}
    }
    let (world,topo,attempt)=selected.with_context(||format!("style {} seed {} topology did not pass in 12 deterministic candidates: {}",request.style,request.seed,serde_json::to_string(&attempts).unwrap()))?;
    let output=PathBuf::from(&arguments[1]);fs::create_dir_all(&output)?;
    let mut raw=BufWriter::new(fs::File::create(output.join("height.f32"))?);
    let mut ocean=BufWriter::new(fs::File::create(output.join("ocean.u8"))?);
    for y in 0..request.height {
        let lat=std::f64::consts::FRAC_PI_2-(y as f64+0.5)/request.height as f64*std::f64::consts::PI;
        for x in 0..request.width {
            let lon=(x as f64+0.5)/request.width as f64*std::f64::consts::TAU-std::f64::consts::PI;
            let p=[lat.cos()*lon.cos(),lat.sin(),lat.cos()*lon.sin()];
            raw.write_all(&coarse_height(&world,p).to_le_bytes())?;
            ocean.write_all(&[u8::from(world.cells[geometry::direction_index(p,world.config.resolution)].ocean)])?;
        }
    }
    raw.flush()?; ocean.flush()?;
    let receipt=serde_json::json!({"requested_seed_u64":request.seed.to_string(),"style":request.style,
        "selected_attempt":attempt,"selected_seed_u64":world.config.seed.to_string(),
        "generator_version":GENERATOR_VERSION,"native_world_id":world.id,"native_config":world.config,
        "native_stats":world.stats,"canonical_height_topology":topo,"attempts":attempts,
        "raster":{"width":request.width,"height":request.height,"dtype":"<f4","projection":"equirectangular","samples":"pixel-centres","y_axis":"south","sea_level_m":0.0,"reconstruction":"bilinear-face-centres-with-continuous-one-cell-overlap","longitude_axis":"atan2(z,x)","sea_convention":"height<0; inland below-datum basins included","physical_ocean_mask":"nearest-native-cell; diagnostic only"},
        "generation_seconds":started.elapsed().as_secs_f64()});
    fs::write(output.join("metadata.json"),serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}",serde_json::to_string(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_has_no_face_edge_or_corner_jump() {
        let mut world=World::generate(WorldConfig { resolution:16,erosion_steps:0,..Default::default() }).unwrap();
        // Strongly differing adjacent cells expose face-switch discontinuities.
        for (index,cell) in world.cells.iter_mut().enumerate() {
            cell.height_m=((index*179)%1009) as f32;
        }
        for a in 0..3 {
            for b in a+1..3 {
                for sa in [-1.,1.] {
                    for sb in [-1.,1.] {
                        for t in [-1.,-0.83,-0.25,0.,0.39,0.92,1.] {
                            let c=3-a-b;
                            let mut p=[0.;3];p[a]=sa;p[b]=sb;p[c]=t;
                            let mut left=p;left[a]*=1.+1e-9;
                            let mut right=p;right[b]*=1.+1e-9;
                            assert!((coarse_height(&world,geometry::normalize(left))-coarse_height(&world,geometry::normalize(right))).abs()<0.001,"edge {p:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reconstruction_preserves_constant_and_cell_centres() {
        let mut world=World::generate(WorldConfig { resolution:16,erosion_steps:0,..Default::default() }).unwrap();
        for cell in &mut world.cells {cell.height_m=1234.5;}
        for p in [[1.,1.,1.],[-1.,1.,0.7],[1.,0.,0.],[0.,1.,0.]] {
            assert_eq!(coarse_height(&world,geometry::normalize(p)),1234.5);
        }
        for (index,cell) in world.cells.iter_mut().enumerate() {cell.height_m=index as f32;}
        for face in 0..6u8 {
            for (x,y) in [(4,4),(8,8),(12,11)] {
                let p=geometry::direction(face,2.*(x as f64+0.5)/16.-1.,2.*(y as f64+0.5)/16.-1.);
                assert_eq!(coarse_height(&world,p),world.cells[geometry::cell_index(face,x,y,16)].height_m);
            }
        }
    }
}
