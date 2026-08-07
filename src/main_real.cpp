/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Runs the controller against the real G1 over DDS.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi */

#include <iostream>

#include "g1_sim2real/unitree_comm.hpp"
#include "g1_sim2real/wholebody_rl.hpp"

int main(int argc, char** argv)
{
  const std::string config_path = (argc > 1) ? argv[1] : "../configs/g1_sim2real.yaml";

  try
  {
    WholeBodyRL controller(config_path);
    UnitreeComm comm(controller, controller.network_interface());
    comm.Run();
  }
  catch (const std::exception& e)
  {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
