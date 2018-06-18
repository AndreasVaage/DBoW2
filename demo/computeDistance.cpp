/**
 * File: testVoc.cpp

 */

#include <fstream> 
#include <iostream>
#include <vector>

#include <stdio.h>
#include <math.h>
#include <limits>

#include "matrix.h"

using namespace std;

int MATCHING_IMAGE =  70;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
vector<Matrix> loadPoses(string file_name) {
  vector<Matrix> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
vector<float> trajectoryDistances (vector<Matrix> &poses) {
  vector<float> dist;
  Matrix P1 = poses[MATCHING_IMAGE];
  for (int32_t i=0; i<poses.size(); i++) {
    Matrix P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if(argc != 3)
    {
        cerr << endl << "Usage: ./computeDistance path_to_ground_trouth output_name" << endl;
        return 1;
    }

  const std::string gt_filename = string(argv[1]);
  const std::string outputName = string(argv[2]);

  vector<Matrix> poses_gt     = loadPoses(gt_filename);
  vector<float> dist = trajectoryDistances(poses_gt);

  ofstream file(outputName + ".dat");
  file << "#frame dist" << endl;
  for (int32_t i=0; i<dist.size(); i++)
  {
    file << i << ' ' << dist[i] << endl;
  }
  file.close();
  // ----------------------------------------------------------------------------
}
