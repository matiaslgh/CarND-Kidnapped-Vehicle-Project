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

Particle createParticle(int id, double x, double y, double theta) {
  Particle particle;
  particle.id = id;
  particle.weight = 1;
  particle.x = x;
  particle.y = y;
  particle.theta = theta;
  return particle;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 80;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine generator;

  for (int i = 0; i < num_particles; i++){
    Particle particle = createParticle(i, dist_x(generator), dist_y(generator), dist_theta(generator));
    particles.push_back(particle);
  }

  is_initialized = true;
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

    if (fabs(yaw_rate) < 0.0001) {
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
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs &observation = observations[i];
    double min_distance = std::numeric_limits<double>::max();
    int closest_landmark_id;
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs &prediction = predicted[j];
      
      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (distance < min_distance) {
        min_distance = distance;
        closest_landmark_id = prediction.id;
      }
    }
    observation.id = closest_landmark_id;
  }
}

vector<LandmarkObs> transformCarToMapCoordinates(const vector<LandmarkObs> &observations, Particle particle) {
  vector<LandmarkObs> transformed_observations;
  double theta = particle.theta;

  for (int i=0; i < observations.size(); i++){
    LandmarkObs observation = observations[i];
    LandmarkObs transformed_obervation;

    transformed_obervation.x = observation.x * cos(theta) - observation.y * sin(theta) + particle.x;
    transformed_obervation.y = observation.x * sin(theta) + observation.y * cos(theta) + particle.y;
    transformed_obervation.id = observation.id;

    transformed_observations.push_back(transformed_obervation);
  }

  return transformed_observations;
}

vector<LandmarkObs> filterOutLandmarksOutOfSensorRange(double sensor_range,
                                                      Particle particle,
                                                      vector<Map::single_landmark_s> landmark_list) {
  vector<LandmarkObs> predicted_landmarks;
  for (int i = 0; i < landmark_list.size(); i++) {
    Map::single_landmark_s landmark = landmark_list[i];
    double landmark_dist = dist(particle.x, particle.y, landmark.x_f, landmark.y_f );
    if (landmark_dist < sensor_range){
      LandmarkObs predicted_landmark;
      predicted_landmark.id = landmark.id_i;
      predicted_landmark.x = landmark.x_f;
      predicted_landmark.y = landmark.y_f;

      predicted_landmarks.push_back(predicted_landmark);
    }
  }
  return predicted_landmarks;
}

double multivariateGaussianDist(double sigma_x, double sigma_y, double x_obs,
                                double y_obs, double x_pred, double y_pred) {
  double gauss_normalization = 1 / (2 * M_PI * sigma_x * sigma_y);
  double x_term = pow(x_pred - x_obs, 2) / (2 * pow(sigma_x, 2));
  double y_term = pow(y_pred - y_obs, 2) / (2 * pow(sigma_y, 2));
  double exponent = x_term + y_term;
  double result = gauss_normalization * exp(-exponent);
  if (result == 0) {
    return 0.00001; // Avoid returning zero to prevent making weight zero --> weight *= multivariateGaussianDist(...)
  } else {
    return result;
  }
}

double calculateWeight(vector<LandmarkObs> predicted_landmarks, vector<LandmarkObs> observations, double std_landmark[]) {
  double weight = 1.0;
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs &obs = observations[i];
    for (int j = 0; j < predicted_landmarks.size(); j++) {
      LandmarkObs &pred = predicted_landmarks[j];
      // TODO: Use maps instead of arrays to simplify this logic
      if (obs.id == pred.id) {
        weight *= multivariateGaussianDist(sigma_x, sigma_y, obs.x, obs.y, pred.x, pred.y);
      }
    }
  }
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  for (int i=0; i < particles.size(); i++) {
    Particle &particle = particles[i];
    vector<LandmarkObs> predicted_landmarks = filterOutLandmarksOutOfSensorRange(sensor_range, particle, map_landmarks.landmark_list);
    vector<LandmarkObs> transformed_observations = transformCarToMapCoordinates(observations, particle);
    dataAssociation(predicted_landmarks, transformed_observations);
    particle.weight = calculateWeight(predicted_landmarks, transformed_observations, std_landmark); 
  }
}

vector<double> getWeights(vector<Particle> particles) {
  vector<double> weights;
  for (int i = 0; i < particles.size(); i++) {
    weights.push_back(particles[i].weight);
  }
  return weights;
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.

  std::default_random_engine generator;
  vector<Particle> resampled_particles;
  vector<double> weights = getWeights(particles);

  double twice_max_weight = 2.0 * *max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> random_weight(0.0, twice_max_weight);
  std::uniform_int_distribution<int> particle_index(0, num_particles - 1);

  int current_index = particle_index(generator);
  double beta = 0.0;

  for (int i = 0; i < particles.size(); i++) {
    beta += random_weight(generator);
    while (beta > weights[current_index]) {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[current_index]);
  }
  particles = resampled_particles;
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