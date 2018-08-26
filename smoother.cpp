//--------------------------------------------------------------------------------------
// NUMECA International s.a.
// Implementator : D. Gutzwiller                                               Apr. 2018
//
// Description : Simple additive smoothing / blur filter for a RGBA png file. 
//               Implemented with OpenMP shared memory parallelism. 
//
// Assignment tasks:
// 1) Optimize the current shared memory implementation.  Describe what you changed and why.
// 2) Adapt the code for distributed parallelism with MPI.
// 3) Adapt the code for GPU execution with CUDA or OpenACC.
// 4) Implement heterogeneous MPI+CUDA/OpenACC support. 
//
// Constraints:
// 1) All versions of the code should yield identical results.
// 2) A blur stencil up to 4 pixels wide should be supported (see variable "range")
//
//--------------------------------------------------------------------------------------

#include <cmath>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <png.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

//--------------------------------------------------------------------------------------
// Class for reading/writing a png image from disk.  
// Also contains a method for blurring the image. 
//--------------------------------------------------------------------------------------
class ImagePNG
{
public:

  ImagePNG()
  { 
    _nbColor   = 4; // hard coded for RGBA
    _width     = 0;
    _height    = 0;
    _allocated = false;
  }

  ~ImagePNG()
  {
    if (_allocated)
    {
      for (int y=0; y<_height; y++) 
      {  
        free(_rowP[y]);
      }
      free(_rowP);
    } 
  }

  // smooths the image
  // nbSteps: number of smoothing steps to apply.
  // range: maximum extent of neighboring cells to be included.
  //        smoothedvalue = (sum of all cell values in x/y +- range) / (nb cells in stencil)
  //        near the image boundaries only the internal neighbors are used.  
  void smooth(int nbSteps, int range)
  {
    if (!_allocated)
    {
      std::cout << "[ImagePNG] data not allocated, can't smooth " << std::endl;
      abort();
    }
    std::cout << "[ImagePNG] smoothing with " << nbSteps << " steps and a stencil range of " << range << std::endl;

    // temporary array for smoothed RGBA values (store as integer for simplicity)
    // we use a temporary array for thread safety
    int** smoothed = new int*[_height];
    for (int y=0; y<_height; y++)
    {
      smoothed[y] = new int[_nbColor*_width];
    }

    for (int iStep=0; iStep<nbSteps; iStep++)
    {
      auto timeStart = Clock::now();
      std::cout << "[ImagePNG] smoothing step " << iStep << " started " << std::endl;

      // zero out temporary data
      #pragma omp parallel for
      for (int y=0; y<_height; y++)
      {
        for (int x=0; x<_width; x++)
        {
          for (int c=0; c<_nbColor; c++)
          {
            smoothed[y][x*_nbColor+c] = 0;
          }
        }
      }
    
      #pragma omp parallel for  
      for (int y=0; y<_height; y++)
      {
        int yOffMin = -std::min(range,y); 
        int yOffMax = std::min(range,(_height-y-1));
        for (int x=0; x<_width; x++)
        {
          int xOffMin = -std::min(range,x);
          int xOffMax = std::min(range,(_width-x-1));
          int nbAdd   = 0; 
          for (int yOff=yOffMin; yOff<=yOffMax; yOff++)
          {
            for (int xOff=xOffMin; xOff<=xOffMax; xOff++)
            { 
              nbAdd++;
              for (int c=0; c<_nbColor; c++)
              { 
                smoothed[y][x*_nbColor+c] += _rowP[y+yOff][(x+xOff)*_nbColor+c];
              } 
            }
          }
          for (int c=0; c<_nbColor; c++)
          {
            smoothed[y][x*_nbColor+c] -= _rowP[y][x*_nbColor+c];
            smoothed[y][x*_nbColor+c] /= (nbAdd-1);
          }
        }
      }

      #pragma omp parallel for
      for (int y=1; y<_height-1; y++)
      {
        for (int x=1; x<_width-1; x++)
        {
          for (int c=0; c<_nbColor; c++)
          {
            _rowP[y][x*_nbColor+c] = smoothed[y][x*_nbColor+c];
          }
        }
      }

      auto timeEnd = Clock::now();
      std::chrono::duration<double> elapsed = timeEnd-timeStart;
      std::cout << "[ImagePNG] smoothing step done in " << elapsed.count() << " seconds." << std::endl;
    }

    // clear the temporary data
    for (int y=0; y<_height; y++)
    {
      delete[] smoothed[y];
    }
    delete[] smoothed;
  }

  void readFromFile(char* fileIn)
  {
    // read the header
    unsigned char header[8];
    FILE *fp = fopen(fileIn,"rb");
    if (!fp)
    {
      std::cout << "[ImagePNG] error reading file " << fileIn << std::endl;
      abort();
    }
    fread(header,1,8,fp);
    if (png_sig_cmp(header,0,8))
    {  
      std::cout << "[ImagePNG] error reading header " << fileIn << std::endl;
      abort();
    }
    
    // initialize - warning, limited error checking here
    _pngP = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); 
    if (!_pngP)
    {
      std::cout << "[ImagePNG] error creating struct" << std::endl;
      abort();
    }
    _infoP = png_create_info_struct(_pngP);
    setjmp(png_jmpbuf(_pngP));
    png_init_io(_pngP, fp);
    png_set_sig_bytes(_pngP, 8);
    png_read_info(_pngP, _infoP);
    _width  = png_get_image_width(_pngP, _infoP);
    _height = png_get_image_height(_pngP, _infoP);
    _color  = png_get_color_type(_pngP, _infoP);
    _depth  = png_get_bit_depth(_pngP, _infoP);
    png_read_update_info(_pngP, _infoP);
    std::cout << "[ImagePNG] read, image has size  [" << _width << "," << _height << "]" << std::endl;
    if (PNG_COLOR_TYPE_RGBA!=_color)
    {
      std::cout << "[ImagePNG] only RGBA supported" << std::endl;
      abort();
    }

    // read file
    if (setjmp(png_jmpbuf(_pngP)))
    {
      std:: cout << "[ImagePNG] error reading image" << std::endl;
    }
    _rowP = (png_bytep*) malloc(sizeof(png_bytep) * _height);
    for (int y=0; y<_height; y++)
    {
      _rowP[y] = (png_byte*) malloc(png_get_rowbytes(_pngP,_infoP));
    }
    png_read_image(_pngP, _rowP);
    fclose(fp);

    // set flag indicated the data is allocated
    _allocated = true;
  }

  void writeToFile(char* fileOut)
  {
    FILE *fp = fopen(fileOut,"wb");
    if (!fp)
    {
      std::cout << "[ImagePNG] error opening file " << fileOut << std::endl;
      abort();
    }

    // initialize, limited error checking
    _pngP  = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    _infoP = png_create_info_struct(_pngP);
    setjmp(png_jmpbuf(_pngP));
    png_init_io(_pngP,fp); 

    // write header
    setjmp(png_jmpbuf(_pngP));
    png_set_IHDR(_pngP,_infoP,_width,_height,_depth,_color,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_BASE,PNG_FILTER_TYPE_BASE);
    png_write_info(_pngP,_infoP);

    // write data
    setjmp(png_jmpbuf(_pngP));
    png_write_image(_pngP, _rowP);

    // finish
    setjmp(png_jmpbuf(_pngP));
    png_write_end(_pngP,NULL);
    fclose(fp);
  }

private: 
    bool        _allocated;
    int         _width;
    int         _height; 
    int         _nbColor;
    png_structp _pngP;
    png_infop   _infoP;
    png_byte    _color;
    png_byte    _depth;
    png_bytep*  _rowP;
};

// main program.  
// instantiates a png reader on the heap and applies the smoothing. 
int main(int argc, char** argv) 
{
  std::cout << "[PNG Smoother] >> IN " << std::endl;
  auto timeStart = Clock::now();

  char *fileIn;
  char *fileOut;
  int  nbSteps,range;
  if (argc == 5)
  {
    fileIn  = argv[1];
    fileOut = argv[2];
    nbSteps = atoi(argv[3]);
    range   = atoi(argv[4]);
  }
  else
  {
    std::cout << "[PNG Smoother] Incorrect number of arguments" << std::endl;
    std::cout << "[PNG Smoother] ./pngSmoother <filein.jpg> <fileout.jpg> <nbSteps> <smoothingRange>" << std::endl;
    return 1;
  }
 
  std::cout << "[PNG Smoother] Input   : " << fileIn  << std::endl; 
  std::cout << "[PNG Smoother] Output  : " << fileOut << std::endl;  
  std::cout << "[PNG Smoother] nbSteps : " << nbSteps << std::endl;
  std::cout << "[PNG Smoother] range   : " << range   << std::endl;
 
  // read and smooth
  ImagePNG* theImage = new ImagePNG();
  theImage->readFromFile(fileIn);
  theImage->smooth(nbSteps,range);
  theImage->writeToFile(fileOut);
  delete theImage;

  auto timeEnd = Clock::now();
  std::chrono::duration<double> elapsed = timeEnd-timeStart;
  std::cout << "[PNG Smoother] << OUT, elapsed time " << elapsed.count()  << " seconds." << std::endl; 
}
