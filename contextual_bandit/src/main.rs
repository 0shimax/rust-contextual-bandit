extern crate rusty_machine as rm;
extern crate rand;

use rm::linalg::{Matrix, BaseMatrix};
use rand::{Rng, ThreadRng};
use std::collections::HashMap;


fn generate_arm(content_id: &usize, arms: &HashMap<usize, Arm>, feature_dim: &usize) -> Arm {
    let arm: Arm;
    if !arms.contains_key(content_id) {
        arm = Arm::new(*feature_dim, *content_id, 0.0001, 0.0);
    }else{
        arm = arms.get(content_id).unwrap().clone();
    };
    arm
}

#[derive(Debug, Clone)]
struct Arm{
    content_id : usize,
    alpha : f64,
    norm_mean : Matrix<f64>,
    cov_matrix: Matrix<f64>,
    win_rate : f64,
    win : f64,
    lose : f64,
}

impl Arm {
    fn new(feature_dim: usize, content_id: usize, alpha: f64, win_rate: f64) -> Arm {
        Arm {
            content_id: content_id,
            norm_mean: Matrix::zeros(1, feature_dim),
            cov_matrix: Matrix::ones(1, feature_dim),
            alpha: alpha,
            win_rate: win_rate,
            win: 0.0,
            lose: 0.0,
        }
    }

    fn update(&mut self, features: Matrix<f64>, is_click: bool) {
        // error of diag
        self.cov_matrix += Matrix::new(self.cov_matrix.rows(),
                                        self.cov_matrix.cols(),
                                        (&features.transpose()*&features).diag());
        let cost_of_click = is_click as i8 as f64;

        let feat_mul_cost_vec = features.iter().map(|x| cost_of_click*x).collect::<Vec<f64>>();
        let feat_mul_cost = Matrix::new(features.rows(), features.cols(), feat_mul_cost_vec);
        self.norm_mean += feat_mul_cost;

        if is_click{
            self.win += 1.;
        } else{
            self.lose += 1.;
        }
        self.win_rate = &self.win/(&self.win + &self.lose);
    }

    fn predict(&mut self, features: &Matrix<f64>) -> f64 {
        let one_div_cov_vec = self.cov_matrix.iter().map(|x| 1.0/x).collect::<Vec<f64>>();
        let one_div_cov = Matrix::new(self.cov_matrix.rows(),
                                      self.cov_matrix.cols(), one_div_cov_vec);
        // Since the covariance matrix preserves only the diagonal components,
        // it suffices to take the inverse matrix
        let theta = &one_div_cov.elemul(&self.norm_mean);
        // Again, the inverse matrix of the covariance matrix
        // is ​​computed by taking reciprocal
        let mut tmp: f64 = ((features.elemul(&one_div_cov))*features.transpose()).data()[0];
        tmp = &tmp.sqrt()*&self.alpha;
        (theta*features.transpose()).data()[0] + &tmp
    }

    fn print_result(&mut self) {
        println!("content_id:{}, total_num:{}, win_rate:{}",
                &self.content_id, &self.win+&self.lose, &self.win_rate);
    }
}

struct Viewer{
    gender: String,
    rng: ThreadRng,
}

impl Viewer {
    fn new(gender: String) -> Viewer{
        Viewer{
            gender: gender,
            rng: rand::thread_rng(),
        }
    }

    fn view(&mut self, content_id: &usize) -> bool{
        if &self.gender=="man" {
            // Men are easy to click on ads with id 5 or less
            if *content_id < 6 {
                return Some(self.rng.next_f32()).and_then(|n| if n>0.3 {Some(true)} else {Some(false)}).unwrap();
            } else{
                return Some(self.rng.next_f32()).and_then(|n| if n>0.7 {Some(true)} else {Some(false)}).unwrap();
            }
        } else {
            // Women are easy to click on ads with id 6 or higher
            if *content_id > 5{
                return Some(self.rng.next_f32()).and_then(|n| if n>0.3 {Some(true)} else {Some(false)}).unwrap();
            } else {
                return Some(self.rng.next_f32()).and_then(|n| if n>0.7 {Some(true)} else {Some(false)}).unwrap();
            }
        }
    }
}

struct Rulet{
    rng: ThreadRng,
}

impl Rulet{
    fn new() -> Rulet {
        Rulet{
            rng: rand::thread_rng(),
        }
    }

    fn generate_features(&mut self, viewer: &Viewer) -> Matrix<f64> {
        let features = Some(&viewer.gender)
            .and_then(|gender|
                if gender=="man" {Some(Matrix::new(1,2,vec![1.,0.]))}
                else {Some(Matrix::new(1,2,vec![0.,1.]))}
            ).unwrap();
        features
    }

    fn generate_content(&mut self) -> usize{
        self.rng.gen_range(0, 10)
    }

    fn generate_gender(&mut self) -> String {
        if self.rng.next_f32() > 0.5{
            return "man".to_string();
        } else{
            return "women".to_string();
        }
    }
}


fn main() {
    /*Context is for men and women only
    Men are easy to click on ads with id 5 or less
    Women are easy to click on ads with id 6 or higher
    */
    let feature_dim = 2;
    let num_of_views = 10000;
    let mut rulet = Rulet::new();

    let mut content_id: usize;
    let mut features: Matrix<f64>;
    let mut is_clicked: bool;
    let mut arms: HashMap<usize, Arm> = HashMap::new();

    for step in 0..num_of_views {
        let mut viewer = Viewer::new(rulet.generate_gender());

        content_id = rulet.generate_content();
        features = rulet.generate_features(&viewer);
        is_clicked = viewer.view(&content_id);

        let mut arm = generate_arm(&content_id, &arms, &feature_dim);
        arm.update(features, is_clicked);
        arms.remove(&content_id);
        arms.insert(content_id, arm.clone());
    }

    let man_mat: Matrix<f64> = Matrix::new(1,2, vec![1.,0.]);
    let woman_mat: Matrix<f64> = Matrix::new(1,2, vec![0.,1.]);

    println!("print result======");
    for (_, arm) in arms.iter() {
        let mut arm = arm.clone();  // impl needs mutable self
        &arm.print_result();
        println!("Click rate when men browse: {}", &arm.predict(&man_mat) );
        println!("Click rate when women browse: {}", &arm.predict(&woman_mat) );
    }

}
