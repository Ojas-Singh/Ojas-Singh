#![allow(non_snake_case)]
use std::env;
extern crate num_iter;
extern crate bit_vec;
// extern crate ndarray_linalg;
use bit_vec::BitVec;
use std::fs::File;
use std::io::{self, BufRead};
// use std::io::prelude::*;
use std::path::Path;
// use std::convert::TryFrom;
// use ndarray_linalg::eigh::EigValsh;
// use num::complex::Complex;

fn help() {
    println!("usage:: pass system args as :
n , m , excite , oneElectronFilename , twoElectronFilename ");
}

fn main() {
    println!("Reading Files ...");
    let args: Vec<String> = env::args().collect();
    let n0 = &args[1] ;
    let m0 = &args[2] ;
    let excite = &args[3] ;
    let f1 = &args[4] ;
    let f2 = &args[5] ;
    let n: isize = match n0.parse() {
        Ok(n0) => {
            n0
        },
        Err(_) => {
            eprintln!("error: second argument not an integer");
            help();
            return;
        },
    };
    let m: isize = match m0.parse() {
        Ok(m0) => {
            m0
        },
        Err(_) => {
            eprintln!("error: second argument not an integer");
            help();
            return;
        },
    };
    println!("n : {}, m : {}, excite : {}, oneElectron : {}, twoElectron : {}", n,m,excite,f1,f2);
    let Honemat = Hone(f1.to_string(),m as usize);
    let Vmat = Vpqrs(f2.to_string(), m as usize);
    let binstates = createslaterdeterminants(n as usize, m as usize, excite.to_string());
    println!("Total Generated States :{}",binstates.len());
    // let x = binstates[1].clone();
    // addparticle(3, x.clone());
    // println!("{:?}",binstates);
    // println!("{:?}",Honemat);
    // println!("{:?}",Vmat);
    // let mut a = BitVec::from_elem(10, false);
    // let mut b = BitVec::from_elem(10, false);
    // a.set(0,true);
    // b.set(0,true);
    // // b.set(2,true);
    // println!("{:?} \n\n{:?}",a,b);
    // // let comp = a.and(&b);
    // // println!("{:?}",comp);
    // removeparticle(0, &mut a);
    // println!("{:?} \n\n{:?}",a,b);
    // let p = secondQuantizationOneBodyOperator(1, 2*1+1, binstates[0].clone(), binstates[3].clone());
    // print!("{}\n",p);

    
    let ham = computeHamiltonianMatrix(binstates, Vmat, Honemat, m as usize);
    // let eigen = EigValshInto();
    // print!("Done! {:?}",eigen);


}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn Hone(f1:String,M:usize) -> Vec<Vec<f64>>{
    let mut Hvec  = vec![vec![0.0;M];M];
    let mut line = 0;
    let mut Hstring = vec![];
    let mut Hab =Vec::new() ;
    if let Ok(lines) = read_lines(f1) {
        for line in lines {
            if let Ok(ip) = line {
                Hstring.push(ip);
            }
        }
    }
    for i in Hstring{
        let mut k = i.replace("(","").replace(")", "").replace(" ", "");
        let vec: Vec<&str> = k.split(",").collect();
        let mut a: f64 = vec[0].parse().unwrap();
        let mut b: f64 = vec[1].parse().unwrap();
        // println!("{:?}",vec);
        Hab.push([a,b]);
                                 
    }
    for i in 0..M {
        for j in 0..M {
            Hvec[i][j] = Hab[line][0];
            line +=1;
        }
    }
    return Hvec;
}

pub fn Vpqrs(f2:String,M:usize) -> Vec<Vec<Vec<Vec<f64>>>>{
    let mut Vvec  = vec![vec![vec![vec![0.0;M];M];M];M];
    let mut line = 0;
    let mut Vstring = vec![];
    let mut Vab =Vec::new() ;
    if let Ok(lines) = read_lines(f2) {
        for line in lines {
            if let Ok(ip) = line {
                Vstring.push(ip);
            }
        }
    }
    for i in Vstring{
        let mut k = i.replace("(","").replace(")", "").replace(" ", "");
        let vec: Vec<&str> = k.split(",").collect();
        let mut a: f64 = vec[0].parse().unwrap();
        let mut b: f64 = vec[1].parse().unwrap();
        // println!("{:?}",vec);
        Vab.push([a,b]);
                                 
    }
    for i in 0..M {
        for j in 0..M {
            for k in 0..M {
                for l in 0..M {
                    Vvec[i][j][k][l] = Vab[line][0];
                    line +=1;
                }
            }
        }
    }
    return Vvec;
}

pub fn creatinitialstate(excite:String,n:usize) -> Vec<Vec<isize>> {
    let mut stateout = Vec::new();
    if excite == "Singlet" {
        if n%2 == 0 {
            let mut stateup = Vec::new();
            let mut statedown = Vec::new();
            for i in num_iter::range(0,n/2){
                stateup.push((i+1) as isize);
                statedown.push((i+1) as isize);
            }
            stateout.push(stateup);
            stateout.push(statedown);
            return stateout;
        }
        else {
            let mut stateup = Vec::new();
            let mut statedown = Vec::new();
            for i in num_iter::range(0,n/2){
                stateup.push((i+1) as isize);
                statedown.push((i+1) as isize);
            }
            stateup.push((n/2 +1) as isize);
            stateout.push(stateup);
            stateout.push(statedown);
            return stateout;
        }
    }
    if excite == "Triplet" {  //Left for Future!
        if n%2 == 0 {
            let mut stateup = Vec::new();
            let mut statedown = Vec::new();
            for i in num_iter::range(0,n/2){
                stateup.push((i+1) as isize);
                statedown.push((i+1) as isize);
            }
            stateout.push(stateup);
            stateout.push(statedown);
            return stateout;
        }
        else {
            let mut stateup = Vec::new();
            let mut statedown = Vec::new();
            for i in num_iter::range(0,n/2){
                stateup.push((i+1) as isize);
                statedown.push((i+1) as isize);
            }
            stateup.push((n/2 +1) as isize);
            stateout.push(stateup);
            stateout.push(statedown);
            return stateout;
        }
    }
    return stateout;
}

pub fn odometer(mut state:Vec<isize>,n:isize,m:isize) -> Vec<isize> {
    let mut newstate = state;
    for j in num_iter::range_step(n-1, -1,  -1)  {
        if newstate[j as usize] < m + 1- n+ j {
            let l = newstate[j as usize];
            for k in num_iter::range(j,n) {
                newstate[k as usize] = l + 1 + k - j;
            }
            if newstate[j as usize] != l {
                return newstate
            }
        }
    }
    newstate.iter_mut().for_each(|x| *x = 0);
    return newstate;

}


pub fn createbinarystatevec(state:Vec<isize>,m: usize) -> BitVec {

    let mut binstate = BitVec::from_elem(m*2+1, false);
    for i in state {
        let k:usize = (i-1) as usize; 
        binstate.set(k, true);
    }
    return binstate;
}

pub fn createslaterdeterminants(n:usize,m:usize,excite:String) -> Vec<BitVec> {
    let mut binstates = Vec::new();
    let mut N :usize;
    if n%2 ==0 {
        N = n/2;
    }
    else{
        N = n/2 +1 ;
    }
    let mut stateup = creatinitialstate(excite.to_string(), n as usize)[0].clone();
    let mut statedown = creatinitialstate(excite.to_string(), n as usize)[1].clone();
    let mut statesup = Vec::new();
    statesup.push(stateup.clone());
    let mut statesdown = Vec::new();
    statesdown.push(statedown.clone());
    let mut up =true ;
    let mut down = true ;
    while up {
        stateup = odometer(stateup, N as isize, m as isize) ;
        let sm:isize = stateup.iter().sum();
        if sm == 0  {
            up = false;
        }
        else {
            statesup.push(stateup.clone());
        }
    }
    while down {
        statedown = odometer(statedown, (n/2) as isize, m as isize) ;
        let sm:isize = statedown.iter().sum();
        if sm == 0  {
            down = false;
        }
        else {
            statesdown.push(statedown.clone());
        }
    }

    for i in statesup {
        for j in &statesdown {
            let binstate = createbinarystatevec(mix(i.to_vec(),j.to_vec()), m as usize);
            binstates.push(binstate);
        }
    }
    return binstates;

}

pub fn mix(state1:Vec<isize>,state2:Vec<isize>) -> Vec<isize>{
    let mut state = Vec::new();
    for i in state1 {
        state.push(2*i-1);
    }
    for i in state2 {
        state.push(2*i);
    }
    return state;
}

pub fn sign(n:usize,binstate:&mut BitVec) -> isize {
    let mut s:isize = 1 ;
    if !binstate[binstate.len()-1]  {
        
        for i in 0..n {
            if binstate[i] != false {
                s *= -1;
            }
        }
        return s;
    }
    return s;
}

pub fn addparticle(n:usize,binstate:&mut BitVec) {
    if !binstate[binstate.len()-1] {
        let mut a = BitVec::from_elem(binstate.len(), false);
        a.set(n,true); 
        let comp = a.and(&binstate);
        if comp {
            binstate.set(n,true);
        }
        else{
            binstate.set(binstate.len()-1,true);
        }

    }
}


pub fn removeparticle(n:usize,binstate:&mut BitVec)  {
    if !binstate[binstate.len()-1]  {
        let mut a = BitVec::from_elem(binstate.len(), false);
        a.set(n,true); 
        let comp = a.and(&binstate);
        if !comp {
            binstate.set(n,false);
        }
        else{
            binstate.set(binstate.len()-1,true);
        }

        
    }
    
}
pub fn secondQuantizationOneBodyOperator(p:usize,q:usize,state1:&mut BitVec,state2:&BitVec) -> isize{
    let mut phase = 1;
    let k =state1.len() -1 ;
    removeparticle(q, state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(q, state1);
    addparticle(p,state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(p, state1);
    if state1.and(&state2){
        phase =0 ;
    }
    return phase;
}

pub fn secondQuantizationTwoBodyOperator(p:usize,q:usize,r:usize,s:usize,state:&BitVec,state2:&BitVec) -> isize{
    let mut phase = 1;
    let mut state1 = state.clone(); 
    let k =state1.len() -1 ;
    removeparticle(r,&mut state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(r,&mut  state1);
    removeparticle(s,&mut  state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(s,&mut  state1);
    addparticle(q,&mut  state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(q,&mut  state1);
    addparticle(p,&mut  state1);
    if state1[k] == true {
        return 0;
    }
    phase *= sign(p,&mut  state1);
    if state1.and(&state2){
        phase = 0
    }
    return phase;
}

pub fn computeHamiltonianMatrix(binstates:Vec<BitVec>,v:Vec<Vec<Vec<Vec<f64>>>>,h:Vec<Vec<f64>>,M:usize)-> Vec<Vec<f64>> {
    let nslater = binstates.len();
    let mut hamiltonian = vec![vec![0.0;nslater];nslater];
    for m in 0..nslater {
        for n in m..nslater {
            for p in 0..M {
                for q in 0..M {
                    let mut phase = 0;
                    phase += secondQuantizationOneBodyOperator(2*p, 2*q,&mut binstates[n].clone(),&binstates[m]);
                    phase += secondQuantizationOneBodyOperator(2*p+1, 2*q+1,&mut  binstates[n].clone(), &binstates[m]);
                    hamiltonian[m][n] += h[p][q]*phase as f64;

                    for r in 0..M {
                        for s in 0..M {
                                phase = 0;
                                phase += secondQuantizationTwoBodyOperator(2*p, 2*q, 2*r, 2*s,&binstates[n], &binstates[m]);
                                phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q+1, 2*r+1, 2*s+1, &binstates[n],&binstates[m]);
                                phase += secondQuantizationTwoBodyOperator(2*p, 2*q+1, 2*r, 2*s+1, &binstates[n], &binstates[m]);
                                phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q, 2*r+1, 2*s, &binstates[n], &binstates[m]);
                                hamiltonian[m][n] += 0.5*v[p][r][q][s]*phase as f64 ;
                            
                        }
                    }
                }
            }
            if m!=n {
                hamiltonian[n][m] = hamiltonian[m][n];
            }
        }
    }
    return hamiltonian;
}