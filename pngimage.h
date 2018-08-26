//
// Created by klaus on 25.08.18.
//

#ifndef NUMECA_PNGIMAGE_H
#define NUMECA_PNGIMAGE_H

#include <iostream>
#include <memory>
#include <png.h>
#include <string>
#include <chrono>
#include "cpp14.h"
#if MPI_ENABLED
#include <mpi.h>
#include <sstream>
#endif

typedef std::chrono::high_resolution_clock Clock;

class PNGImage
{
public:
   PNGImage();

   void readFromFile(const std::string &fileIn);

   void writeToFile(const std::string &fileOut);

#ifdef MPI_ENABLED
   void smooth(int nbSteps, int range, int rank, int nNodes);
#else
   void smooth(int nbSteps, int range);
#endif

private:
   int _width;
   int _height;
   int _nbColor;
   std::unique_ptr<png_byte[]> _data;
   std::unique_ptr<png_bytep[]> _rowP;
   png_structp _pngP;
   png_infop _infoP;
   png_byte _color;
   png_byte _depth;
#if MPI_ENABLED
   inline void smoothMPI(int nbSteps, int range, int rank, int nNodes);
#else
   inline void smoothOMP(int nbSteps, int range);
#endif

};


#endif //NUMECA_PNGIMAGE_H
