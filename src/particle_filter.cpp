/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 80;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(x, std[1]);
  std::normal_distribution<double> dist_theta(x, std[2]);
  std::default_random_engine generator;

  for (int i = 0; i < num_particles; i++){
    Particle particle = createParticle(i, dist_x(generator), dist_y(generator), dist_theta(generator));
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

Particle createParticle(int id, double x, double y, double theta) {
  Particle particle;
  particle.id = id;
  particle.weight = 1;
  particle.x = x;
  particle.y = y;
  particle.theta = theta;
  return particle;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  std::default_random_engine generator;
  
  for (int i = 0; i < num_particles; i++) {   
    double pred_x;
    double pred_y;
    double pred_theta;
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double yaw0 = particles[i].theta;

    if (yaw_rate == 0) {
      pred_theta = yaw0;
      pred_x = x0 + velocity * cos(yaw0) * delta_t;
      pred_y = y0 + velocity * sin(yaw0) * delta_t;
    } else {
      pred_theta = yaw0 + yaw_rate * delta_t;
      pred_x = x0 + (velocity/yaw_rate) * (sin(pred_theta) - sin(yaw0));
      pred_y = y0 + (velocity/yaw_rate) * (cos(yaw0) - cos(pred_theta));
    }
    
    std::normal_distribution<double> dist_x(pred_x, std_pos[0]);
    particles[i].x = dist_x(generator);

    std::normal_distribution<double> dist_y(pred_y, std_pos[1]);
    particles[i].y = dist_y(generator);

    std::normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
    particles[i].theta = dist_theta(generator);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each observed measurement
   *   and assign the observed measurement to this particular landmark.
   */
  double min_distance = dist(observations[0].x, observations[0].y, predicted[0].x, predicted[0].y);
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs observation = observations[i];
    LandmarkObs closest_landmark;
    for (int j = 0; i < predicted.size(); i++) {
      LandmarkObs prediction = predicted[i];
      
      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (distance < min_distance) {
        min_distance = distance;
        closest_landmark = prediction;
      }
    }
    observations[i] = closest_landmark;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}