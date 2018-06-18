/**
 * File: testVoc.cpp

 */

#include <fstream> 
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>
#include "FClass.h"
#include "FCNN.h"
#include "FORB.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>



using namespace DBoW2;
using namespace DUtils;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadOrbFeatures(vector<vector<FORB::TDescriptor > > &features, const string &dexcriptor_filename);
void changeOrbStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void loadCNNFeatures(vector<vector<FCNN::TDescriptor > > &features, const string &dexcriptor_filename);
template<class Tvoc, class TDescriptor>
void testVoc(Tvoc &voc, const vector<vector<TDescriptor > > &features);
template<class Tdb, class TDescriptor>
void testDatabase(Tdb &db, const vector<vector<TDescriptor > > &features, string &outputName);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
//int NIMAGES = 3682; //MH01
//int NIMAGES = 4541; //KITTI 00
int NIMAGES = 2761;   //KITTI 05
int MATCHING_IMAGE =  70;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if(argc != 4 && argc != 5)
    {
        cerr << endl << "Usage: ./testVoc path_to_vocabulary path_to_descriptor_file|path_to_image_folder orb|cnn [output_name]" << endl;
        return 1;
    }

  const std::string voc_filename = string(argv[1]);
  const std::string dexcriptor_filename = string(argv[2]);
  const std::string featureType = string(argv[3]);
  string outputName = "none";
  if(argc == 5)
    outputName = string(argv[4]);

  // ----------------------------------------------------------------------------
  if (featureType == "orb")
  {
    vector<vector<FORB::TDescriptor> > features;
    loadOrbFeatures(features, dexcriptor_filename);
    cout <<endl<< "Loaded "<< features.size() <<" features" << endl;
    cout << "Loading vocabulary" << endl;
    OrbVocabulary voc;
    if (!voc.loadFromTextFile(voc_filename))
    {
      cerr << endl << "Could not load vocabulary from file: " << voc_filename << endl;
      return 1;
    }
    testVoc(voc,features);

    OrbDatabase db(voc, false, 0);
    //voc.clear();
    testDatabase(db,features,outputName);
  }
  // ----------------------------------------------------------------------------
  else if (featureType == "cnn")
  {
    vector<vector<FCNN::TDescriptor> > features;
    loadCNNFeatures(features, dexcriptor_filename);
    cout <<endl<< "Loaded "<< features.size() <<" features" << endl;
    cout << "Loading vocabulary" << endl;
    CNNVocabulary voc;
    if (!voc.loadFromTextFile(voc_filename))
    {
      cerr << endl << "Could not load vocabulary from file: " << voc_filename << endl;
      return 1;
    }
    cout << "Testing vocabulary" << endl;
    testVoc(voc,features);

    CNNDatabase db(voc, false, 0);
    //voc.clear();
    testDatabase(db,features,outputName);
  }
  // ----------------------------------------------------------------------------
  else
  {
    cerr << endl << "Provided last argument: " << featureType << " needs to be orb or cnn" << endl;
    return 1;
  }

  return 0;

}

// ----------------------------------------------------------------------------

void loadCNNFeatures(std::vector<std::vector<FCNN::TDescriptor > > &features, const string &dexcriptor_filename)
{ 
    features.clear();
    //features.reserve(NIMAGES);
    std::cout<<"About to load features of " << NIMAGES<< " images."<<std::endl;

    std::cout << "From file: " << dexcriptor_filename << std::endl;
    // Read from file
    std::ifstream f{dexcriptor_filename, std::ios::binary};
    if (!f) { throw std::runtime_error{std::strerror(errno)}; }

    int nfeatures = 0;
    int nFrames = 0;
    std::cout << "With CNN descriptor size: "<<FCNN::L << std::endl;
    // Size of CNN descriptor is set in FCNN.h
    bool stop = false;

    std::cout << "Loading features..."<< std::endl;
    while( !f.eof() )
    {
      std::vector<FCNN::TDescriptor> vfeaturesFromOneFrame;
      while(!f.eof())
      {

        // Read (x,y)-coordinate of feature
        int x;
        int y;
        f.read(reinterpret_cast<char*>(&x), sizeof(std::int32_t));
        f.read(reinterpret_cast<char*>(&y), sizeof(std::int32_t));

        if ( x==-1 || y==-1)
        {
          break;
        }


        // Read descriptor of feature
        FCNN::TDescriptor descriptor;
        descriptor.reserve(FCNN::L);

        for (auto i = 0; i < FCNN::L; ++i) {
            float t;
            f.read(reinterpret_cast<char *>(&t), sizeof(t));
            if (!f) { throw std::runtime_error{std::strerror(errno)}; }
            descriptor.push_back(t);
        }
        nfeatures +=1;


        // Save feature in vector of features
        vfeaturesFromOneFrame.push_back(descriptor);
      }
      nFrames +=1;
      nfeatures = 0;
      if (!vfeaturesFromOneFrame.empty())
      {
        features.push_back(vfeaturesFromOneFrame);
      }

      // Print status
      if (nFrames % 4000 == 0)
        std::cout <<std::endl<<"["<<nFrames<<"/"<<NIMAGES<<"]";
      if (nFrames % 200 == 0)
        std::cout <<"#"<< std::flush;
    }

    std::cout << "Done"<< std::endl;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void loadOrbFeatures(vector<vector<FORB::TDescriptor > > &features, const string &imagePath)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    //ss << imagePath << "/image" << i << ".png"; MH01
    //ss << imagePath << "/00" << i << ".png";
    ss << imagePath << "/" << std::setw(6) << std::setfill('0') << i << ".png";
    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<FORB::TDescriptor>());
    changeOrbStructure(descriptors, features.back());

    // Print status
    if (i % 400 == 0)
      std::cout <<std::endl<<"["<<i<<"/"<<NIMAGES<<"]";
    if (i % 20 == 0)
      std::cout <<"#"<< std::flush;
  }
}

// ----------------------------------------------------------------------------

void changeOrbStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------
template<class Tvoc,class TDescriptor>
void testVoc(Tvoc &voc, const vector<vector<TDescriptor > > &features)
{
  int NIMAGES = 4;
  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }
}

// ----------------------------------------------------------------------------
template<class Tdb,class TDescriptor>
void testDatabase(Tdb &db, const vector<vector<TDescriptor > > &features, string &outputName)
{
  int _NIMAGES = 4;
  cout << "Creating a small database..." << endl;

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;

  if (outputName == "none")
  {
    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(int i = 0; i < _NIMAGES; i++)
      {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the 
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
      }
  cout << endl;
  }
  else 
  {
    db.query(features[MATCHING_IMAGE], ret, NIMAGES);
    ofstream file(outputName + ".dat");
    file << "#x y" << endl;
    QueryResults::const_iterator rit;
    for(rit = ret.begin(); rit != ret.end(); ++rit)
    {
      file << rit->Id << ' ' << rit->Score << endl;
    }
    file.close();
  }
  

}

// ----------------------------------------------------------------------------

