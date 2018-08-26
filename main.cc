//
// Created by klaus on 25.08.18.
//

#include <iostream>
#ifdef MPI_ENABLED
#include <sstream>
#include <mpi.h>
#endif
#include "pngimage.h"

using namespace std;

int main(int argc, char **argv)
{
   const auto start = Clock::now();

   int rank = 0;
   int nNodes = 1;
   string label = "";
#ifdef MPI_ENABLED
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nNodes);
   stringstream ss;
   ss << ">>> rank " << rank << " ";
   label = ss.str();
#endif

   char *fileIn;
   char *fileOut;
   int nbSteps, range;
   if (argc == 5) {
      fileIn = argv[1];
      fileOut = argv[2];
      nbSteps = atoi(argv[3]);
      range = atoi(argv[4]);
   } else {
      if (rank == 0) {
         cout << label << "[PNG Smoother] Incorrect number of arguments" << endl;
         cout << label << "[PNG Smoother] ./pngSmoother <filein.jpg> <fileout.jpg> <nbSteps> <smoothingRange>" << endl;
      }
      return 1;
   }

   if (rank == 0) {
      cout << label << "[PNG Smoother] Input   : " << fileIn << endl;
      cout << label << "[PNG Smoother] Output  : " << fileOut << endl;
      cout << label << "[PNG Smoother] nbSteps : " << nbSteps << endl;
      cout << label << "[PNG Smoother] range   : " << range << endl;
   }

   PNGImage image;

   if (rank == 0)
      image.readFromFile(fileIn);
#ifdef MPI_ENABLED
   image.smooth(nbSteps, range, rank, nNodes);
#else
   image.smooth(nbSteps, range);
#endif
   if (rank == 0)
      image.writeToFile(fileOut);
#ifdef MPI_ENABLED
   MPI_Finalize();
#endif
   const chrono::duration<double> elapsed = Clock::now() - start;
   cout << label << "[PNG Smoother] << OUT, elapsed time " << elapsed.count() << " seconds." << endl;

   return 0;
}