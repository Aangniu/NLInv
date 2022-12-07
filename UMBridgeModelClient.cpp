#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/UniformBox.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/OneStepCachePiece.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"
#include "MUQ/Modeling/LinearAlgebra/HessianOperator.h"
#include "MUQ/Modeling/LinearAlgebra/StochasticEigenSolver.h"

#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"

#include "MUQ/Modeling/UMBridge/UMBridgeModPieceServer.h"
#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/graph/graphviz.hpp>

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Modeling/UMBridge/json.h"
#include <MUQ/Utilities/HDF5/H5Object.h>

/***
## Overview

The UM-Bridge interface allows coupling model and UQ codes through HTTP. A model may then
be implemented in virtually any programming language or framework, run in a container
or even on a remote machine. Likewise, the model does not make any assumptions on how the client is implemented.

This example shows how to connect to a running UM-Bridge server that is implemented in the UM-Bridge Server example.
The server provides the physical model, while the client is responsible for the UQ side.

The UM-Bridge interface is fully integrated in MUQ and can be used by means of the UMBridgeModPiece class.
Once such an UMBridgeModPiece is set up, it can be used like any other ModPiece. If the model supports the respective
functionality, the ModPiece then provides simple model evaluations,
gradient evaluations, applications of the Jacobian etc.

*/
namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
using json = nlohmann::json;

int main(){
/***
## Connect to model server
First, we set up an UMBridgeModPiece that connects to our model server, giving the server's address
and the model name we'd like to use. This assumes that, before the client is started,
the model server from the UM-Bridge Server example is already running on your machine.
*/
  json configs;
  configs["level"] = 0;
  configs["verbosity"] = false;
  configs["vtk_output"] = false;
  
  std::cout << to_string(configs) << std::endl;
  auto mod = std::make_shared<UMBridgeModPiece>("http://0.0.0.0:4254","forward",configs);

  std::cout << mod->inputSizes << std::endl;
  std::cout << mod->outputSizes << std::endl;

  Eigen::VectorXd input_mu(4);
  input_mu << 1.0,1.0,1.0,1.0;
  std::vector<Eigen::VectorXd> inputs;
  inputs.push_back(input_mu);
  Eigen::MatrixXd input_cov(4,4);
  input_cov << 0.2, 0.0,0.0,0.0,
               0.0, 0.2,0.0,0.0,
               0.0, 0.0,0.2,0.0,
               0.0, 0.0,0.0,0.2;

/* Evaluating and output to a file
// This section can be de-commented for evaluating a single set of model input and checking the code correctness
  std::cout << "Evaluating... \n" << std::endl;
  std::vector<Eigen::VectorXd> outputs = mod->Evaluate(inputs);

  std::string home_dir = std::getenv("HOME");
  std::string finame_o;
  finame_o = home_dir + "/Exahype/Points_out_lya_4input.txt";
  std::ofstream outputsfile;
  
  std::cout << "Output to " << finame_o << "\n";
  outputsfile.open(finame_o, std::ios_base::out);
  outputsfile << std::fixed << std::scientific;
  for(int i=0; i<40; i++){
    outputsfile << outputs[0][i] << "\n";
  }
  outputsfile.close();
  std::cout << "Output of modulus generated!" << std::endl;
*/

  auto graph = std::make_shared<WorkGraph>();

  Eigen::VectorXd obser(40);

  std::string home_dir = std::getenv("HOME");
  std::string finame_in;
  // # Read in observations
  // finame_in = home_dir + "/Feng2019_delay.txt";
  finame_in = home_dir + "/Feng2019_nodelay.txt";
  std::ifstream inputsfile;
  
  std::cout << "Reading from " << finame_in << "\n";
  inputsfile.open(finame_in, std::ios_base::in);
  inputsfile >> std::fixed >> std::scientific;
  for(int i=0; i<40; i++){
    inputsfile >> obser[i];
    obser[i] = -obser[i];
    std::cout << "Read in: " << obser[i] << "\n";
  }
  inputsfile.close();
  
  Eigen::MatrixXd likeli_cov = Eigen::MatrixXd::Constant(40, 40, 0.0);
  likeli_cov.diagonal() =  1e-6*Eigen::VectorXd::LinSpaced(obser.size(),1,1);

  auto likelihood = std::make_shared<Gaussian>(obser,
                                              likeli_cov);
  
  std::pair<double, double> b1 = std::make_pair(0.0,3.0);
  std::pair<double, double> b2 = std::make_pair(0.0,4.0);
  std::pair<double, double> b3 = std::make_pair(0.0,4.0);
  std::pair<double, double> b4 = std::make_pair(0.0,2.0);

  auto prior_Uni = std::make_shared<UniformBox>(b1,b2,b3,b4);

  auto mod_oneStep = OneStepCachePiece(mod);

  auto prior = std::make_shared<Gaussian>(input_mu,input_cov);
  graph->AddNode(std::make_shared<IdentityOperator>(4), "Parameters");
  graph->AddNode(mod, "forward model");
  graph->AddNode(likelihood->AsDensity(), "likelihood");
  graph->AddEdge("forward model", 0, "likelihood", 0);

  graph->AddNode(prior_Uni->AsDensity(), "prior");
  graph->AddNode(std::make_shared<IdentityOperator>(4), "Prior Output");
  graph->AddEdge("Parameters", 0, "prior", 0);
  graph->AddEdge("Parameters", 0, "forward model", 0);

  graph->AddNode(std::make_shared<DensityProduct>(2), "posterior");
  graph->AddEdge("prior", 0, "posterior", 0);
  graph->AddEdge("likelihood", 0, "posterior", 1);

  graph->AddNode(std::make_shared<IdentityOperator>(40), "Model Output");
  graph->AddEdge("forward model", 0, "Model Output", 0);

  
  std::cout << "Generating WorkGraph..." << std::endl;

  graph->Visualize("WorkGraph.pdf");

  Eigen::MatrixXd cov_prop(4,4);
  cov_prop << 0.4, 0.0, 0.0, 0.0,
              0.0, 0.4, 0.0, 0.0,
              0.0, 0.0, 0.1, 0.0,
              0.0, 0.0, 0.0, 0.1;
  

  pt::ptree pt;
  const unsigned int numSamps = 7.0e4;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 0);
  pt.put("PrintLevel",3);
  // pt.put("Prior Node", "prior");
  // pt.put("Likelihood Node", "likelihood");
  pt.put("KernelList", "Kernel1"); // Name of block that defines the transition kernel
  pt.put("Kernel1.Method","MHKernel");  // Name of the transition kernel class
  pt.put("Kernel1.Proposal", "MyProposal"); // Name of block defining the proposal distribution
  pt.put("Kernel1.MyProposal.Method", "AMProposal"); // Name of proposal class
  pt.put("Kernel1.MyProposal.ProposalVariance", 1.0); // Variance of the isotropic MH proposal
  pt.put("Kernel1.MyProposal.AdaptSteps", 100); // Variance of the isotropic MH proposal
  pt.put("Kernel1.MyProposal.AdaptStart", 1000); // Variance of the isotropic MH proposal
  // pt.put("Kernel1.MyProposal.AdaptScale", 1.0); // Variance of the isotropic MH proposal

  auto problem = std::make_shared<SamplingProblem>(graph->CreateModPiece("posterior"),
                                                graph->CreateModPiece("Model Output"));


  Eigen::VectorXd startPt(4);
  startPt << 0.5, 1.5, 1.5, 1.0;
  auto mcmc = MCMCFactory::CreateSingleChain(pt, problem);

  std::shared_ptr<SampleCollection> samps = mcmc->Run(startPt);
  std::shared_ptr<SampleCollection> sampsQOI = mcmc->GetQOIs();

  Eigen::MatrixXd sampsAsMat = samps->AsMatrix();
  Eigen::MatrixXd sampsQOIAsMat = sampsQOI->AsMatrix();

  samps->WriteToFile("./output/samples_Feng2019_lyac_inp4_70000_LWModel_UniPriorW_1em6_PV0p4_3442_c.h5"); // LogTarget
  sampsQOI->WriteToFile("./output/models_Feng2019_lyac_inp4_70000_LWModel_UniPriorW_1em6_PV0p4_3442_c.h5");

  std::string samp_file = "./output/samples_Feng2019_lyac_inp4_70000_LWModel_UniPriorW_1em6_PV0p4_3442_c.txt";
  std::string model_file = "./output/models_Feng2019_lyac_inp4_70000_LWModel_UniPriorW_1em6_PV0p4_3442_c.txt";
  std::ofstream of(samp_file, std::ios::out | std::ios::trunc);
  if(of)  // si l'ouverture a réussi
  {   
    // instructions
    of << "Here is the matrix src:\n" << sampsAsMat.transpose() << "\n";
    of.close();  // on referme le fichier
  }
  else  // sinon
  {
    std::cout << "Erreur à l'ouverture !" << std::endl;
  }

  std::ofstream of2(model_file, std::ios::out | std::ios::trunc);
  if(of2)  // si l'ouverture a réussi
  {   
    // instructions
    of2 << "Here is the matrix src:\n" << sampsQOIAsMat.transpose() << "\n";
    of2.close();  // on referme le fichier
  }
  else  // sinon
  {
    std::cout << "Erreur à l'ouverture !" << std::endl;
  }

  return 0;
}
