//
// Created by klaus on 25.08.18.
//

#include <iostream>
#include <sstream>
#ifdef MPI_ENABLED
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
   stringstream ss;
#ifdef MPI_ENABLED
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nNodes);
   ss << "rank " << rank << " ";
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
         cout << label << "[Error] Incorrect number of arguments" << endl;
         cout << label << "\t" << argv[0] <<" <filein.jpg> <fileout.jpg> <nbSteps> <smoothingRange>" << endl;
      }
      return 1;
   }

   if (rank == 0) {
      cout << label << "[LOG] Input   : " << fileIn << endl;
      cout << label << "[LOG] Output  : " << fileOut << endl;
      cout << label << "[LOG] nbSteps : " << nbSteps << endl;
      cout << label << "[LOG] range   : " << range << endl;
   }

   PNGImage image;
   try {
      if (rank == 0)
         image.readFromFile(fileIn);
#ifdef MPI_ENABLED
      image.smooth(nbSteps, range, rank, nNodes);
#else
      image.smooth(nbSteps, range);
#endif
      if (rank == 0)
         image.writeToFile(fileOut);
   } catch (const string &e) {
      cerr << "[Error] " << e << endl;
#ifdef MPI_ENABLED
      MPI_Finalize();
#endif
      return 1;
   }
#ifdef MPI_ENABLED
   MPI_Finalize();
#endif
   const chrono::duration<double> elapsed = Clock::now() - start;
   ss.str("");
   ss << label << "[LOG] smoothing: done " << elapsed.count() << " s" << endl;
   cout << ss.str();

   return 0;
}