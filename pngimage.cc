//
// Created by klaus on 25.08.18.
//

#include <vector>
#include <sstream>
#include "pngimage.h"

using namespace std;

PNGImage::PNGImage() : _nbColor(4)
{}

void PNGImage::readFromFile(const string &fileIn)
{
   // read the header
   unsigned char header[8];
   FILE *fp = fopen(fileIn.c_str(), "rb");
   if (!fp) {
      stringstream ss;
      ss << "Reading file " << fileIn;
      throw ss.str();
   }
   fread(header, 1, 8, fp);
   if (png_sig_cmp(header, 0, 8)) {
      fclose(fp);
      stringstream ss;
      ss << "Reading header " << fileIn;
      throw ss.str();
   }

   // initialize - warning, limited error checking here
   _pngP = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (!_pngP) {
      fclose(fp);
      throw string("Creating struct");
   }
   _infoP = png_create_info_struct(_pngP);
   setjmp(png_jmpbuf(_pngP));
   png_init_io(_pngP, fp);
   png_set_sig_bytes(_pngP, 8);
   png_read_info(_pngP, _infoP);
   _width = png_get_image_width(_pngP, _infoP);
   _height = png_get_image_height(_pngP, _infoP);
   _color = png_get_color_type(_pngP, _infoP);
   _depth = png_get_bit_depth(_pngP, _infoP);
   png_read_update_info(_pngP, _infoP);
   if (PNG_COLOR_TYPE_RGBA != _color) {
      fclose(fp);
      throw string("Only RGBA supported");
   }
   // read file
   if (setjmp(png_jmpbuf(_pngP))) {
      fclose(fp);
      throw string("Reading image");
   }
   const size_t nRowBytes = png_get_rowbytes(_pngP, _infoP);
   _rowP = make_unique<png_bytep[]>(_height);
   _data = make_unique<png_byte[]>(_height * nRowBytes);
   for (int y = 0; y < _height; y++) {
      png_read_row(_pngP, &_data[y * nRowBytes], NULL);
      _rowP[y] = &_data[y * nRowBytes];
   }
   fclose(fp);
}

void PNGImage::writeToFile(const string &fileOut)
{
   FILE *fp = fopen(fileOut.c_str(), "wb");
   if (!fp) {
      stringstream ss;
      ss << "Opening file " << fileOut;
      throw ss.str();
   }

   // initialize, limited error checking
   _pngP = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   _infoP = png_create_info_struct(_pngP);
   setjmp(png_jmpbuf(_pngP));
   png_init_io(_pngP, fp);

   // write header
   setjmp(png_jmpbuf(_pngP));
   png_set_IHDR(_pngP, _infoP, _width, _height, _depth, _color, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                PNG_FILTER_TYPE_BASE);
   png_write_info(_pngP, _infoP);

   // write data
   setjmp(png_jmpbuf(_pngP));
   png_write_image(_pngP, _rowP.get());

   // finish
   setjmp(png_jmpbuf(_pngP));
   png_write_end(_pngP, NULL);
   fclose(fp);
}

#ifdef MPI_ENABLED
void PNGImage::smooth(int nbSteps, int range, int rank, int nNodes)
{
   smoothMPI(nbSteps, range, rank, nNodes);
#else
void PNGImage::smooth(int nbSteps, int range)
{
   smoothOMP(nbSteps, range);
#endif
}

#ifdef MPI_ENABLED
void PNGImage::smoothMPI(int nbSteps, int range, int rank, int nNodes)
{
   stringstream ss;
   ss << "rank " << rank << " ";
   const string label = ss.str();
   ss.str("");
   ss << label << "[LOG] smoothing: started" << endl;
   cout << ss.str();
   ss.str("");

   /* Broadcast initial data from root */
   MPI_Bcast(&_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&_nbColor, 1, MPI_INT, 0, MPI_COMM_WORLD);

   const int workWidth = _width - 2;
   const size_t  rowLen = _width * _nbColor;
   const size_t workRowLen = workWidth * _nbColor;
   const size_t colLen = _height * _nbColor;
   const size_t nodeSizeSq = nNodes * nNodes;
   vector<MPI_Request> req(nNodes);
   vector<MPI_Status> stat(nNodes);
   int yOffset = (rank == 0) ? 1 : 0;

   vector<int> rowsPerNode(nNodes, _height / nNodes);
   vector<int> rowSendCounts(nNodes, _height / nNodes * rowLen);
   vector<int> rowSendDispl(nNodes);
   for (size_t i = (nNodes - _height % nNodes); i < nNodes; i++) {
      rowsPerNode[i]++;
      rowSendCounts[i] += rowLen;
   }
   const int rowOffset = (rank == 0) ? rowLen : 0;
   int workNumRows = rowsPerNode[rank];
   if (nNodes == 1)
      workNumRows -= 2;
   else if ((rank == (nNodes - 1)) || (rank == 0))
      workNumRows--;

   rowSendDispl[0] = 0;
   for (size_t i = 1; i < nNodes; i++) {
      rowSendDispl[i] = rowSendDispl[i - 1] + rowSendCounts[i - 1];
      if (i <= rank)
         yOffset += rowsPerNode[i - 1];
   }
   vector<int> colsPerNode(nNodes, workWidth / nNodes);
   vector<int> tmpRowSendCountsOfNode(nodeSizeSq);
   vector<int> tmpRowSendDisplOfNode(nodeSizeSq);
   vector<int> colRecvDisplOfNode(nodeSizeSq);
   vector<int> tmpColRecvCountsOfNode(nodeSizeSq);
   vector<int> tmpColSendDisplOfNode(nodeSizeSq);
   for (size_t i = (nNodes - workWidth % nNodes); i < nNodes; i++)
      colsPerNode[i]++;
   for (size_t i = 0; i < nNodes; i++) {
      const bool isRoot = (i == 0);
      const bool isLast = (i == (nNodes - 1));
      for (size_t j = 0; j < nNodes; j++) {
         tmpRowSendCountsOfNode[i * nNodes + j] = colsPerNode[j] * rowsPerNode[i] * _nbColor;
         tmpColRecvCountsOfNode[i * nNodes + j] = tmpRowSendCountsOfNode[i * nNodes + j];
         if (isRoot) {
            tmpColRecvCountsOfNode[j] -= colsPerNode[j] * _nbColor;
            colRecvDisplOfNode[j] = 0;
         } else if (isLast)
            tmpColRecvCountsOfNode[i * nNodes + j] -= colsPerNode[j] * _nbColor;
         if (j == 0) {
            tmpRowSendDisplOfNode[i * nNodes] = 0;
            tmpColSendDisplOfNode[i * nNodes] = 0;
         } else {
            tmpRowSendDisplOfNode[i * nNodes + j] = tmpRowSendDisplOfNode[i * nNodes + j - 1]
                                                     + tmpRowSendCountsOfNode[i * nNodes + j - 1];
            tmpColSendDisplOfNode[i * nNodes + j] = tmpColSendDisplOfNode[i * nNodes + j - 1]
                                                     + tmpColRecvCountsOfNode[i * nNodes + j - 1];
         }
         if (!isRoot)
            colRecvDisplOfNode[i * nNodes + j] = colRecvDisplOfNode[(i - 1) * nNodes + j]
                                                  + tmpRowSendCountsOfNode[(i - 1) * nNodes + j];
      }
   }

   /* Allocate the temporary arrays tmpRow and tmpCol and the
    * work arrays rowData and colData */
   vector<int> tmpRow(rowsPerNode[rank] * workRowLen);
   vector<int> tmpCol(colsPerNode[rank] * colLen);
   vector<int> colData(colsPerNode[rank] * colLen);
   vector<png_byte> rowData(rowSendCounts[rank]);

   /* Scatter the rows each node is responsible of from the root node. */
   if (MPI_Scatterv(&_data[0],
                    &rowSendCounts[0],
                    &rowSendDispl[0],
                    MPI_UNSIGNED_CHAR,
                    &rowData[0],
                    rowSendCounts[rank],
                    MPI_UNSIGNED_CHAR,
                    0,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
      ss.str("");
      ss << label << "Initial scattering failed";
      throw ss.str();
   }

   for (int iStep = 0; iStep < nbSteps; iStep++) {
      const auto timeStart = Clock::now();
      ss.str("");
      ss << label << "[LOG] step " << iStep << ": started" << endl;
      cout << ss.str();

      /* smooth along the x-direction and store the results
       * in a temporary array. The algorithm starts, by
       * calculating the non-normalized value for the first
       * pixel and continues to calculate the values along
       * the x-direction by using the value of the previous
       * pixel. The result will be saved in a column major
       * layout in order to be a continuous block of data for
       * the scattering. */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < rowsPerNode[rank] * _nbColor; i++) {
         const int y = i / _nbColor;
         const int c = i % _nbColor;
         int val = 0;
         for (int x = 0; x <= range; x++)
            val += rowData[y * rowLen + x * _nbColor + c];
         for (int x = 1; x < (_width - 1); x++) {
            const int l = x - range - 1;
            const int r = x + range;
            if (l >= 0)
               val -= rowData[y * rowLen + l * _nbColor + c];
            if (r < _width)
               val += rowData[y * rowLen + r * _nbColor + c];
            tmpRow[(x - 1) * _nbColor * rowsPerNode[rank] + y * _nbColor + c] = val;
         }
      }

      /* Scatter the calculated rows as block of memories to the all
       * other nodes and receive the missing */
      for (int i = 0; i < nNodes; i++)
         MPI_Iscatterv(&tmpRow[0],
                       &tmpRowSendCountsOfNode[i * nNodes],
                       &tmpRowSendDisplOfNode[i * nNodes],
                       MPI_INT,
                       &tmpCol[colRecvDisplOfNode[i * nNodes + rank]],
                       tmpRowSendCountsOfNode[i * nNodes + rank],
                       MPI_INT,
                       i,
                       MPI_COMM_WORLD,
                       &req[i]);
      MPI_Waitall(nNodes, &req[0], &stat[0]);

      /* Store the received column data ordered into the working
       * array colData. */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (size_t i = 0; i < colsPerNode[rank] * nNodes; i++) {
         const size_t icol = i / nNodes;
         const size_t inode = i % nNodes;
         const int nodeColLen = rowsPerNode[inode] * _nbColor;
         const auto srcBegin = &tmpCol[colRecvDisplOfNode[inode * nNodes + rank]
                                       + icol * nodeColLen];
         const auto destBegin = &colData[colLen * icol
                                         + colRecvDisplOfNode[inode * nNodes + rank] / colsPerNode[rank]];
         copy(srcBegin, srcBegin + nodeColLen, destBegin);
      }

      /* Finally smooth along the y-direction. The result is
       * stored directly in the _rowP array. This is thread-safe
       * because each threads writs in its unique memory space.
       * Further, the normilization is done here */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < colsPerNode[rank] * _nbColor; i++) {
         const int x = i / _nbColor;
         const int c = i % _nbColor;
         int val = 0;
         for (int y = 0; y <= range; y++)
            val += colData[x * colLen + y * _nbColor + c];
         for (int y = 1; y < (_height - 1); y++) {
            const int t = y - range - 1;
            const int b = y + range;
            if (t >= 0)
               val -= colData[x * colLen + t * _nbColor + c];
            if (b < _height)
               val += colData[x * colLen + b * _nbColor + c];
            tmpCol[y * colsPerNode[rank] * _nbColor + x * _nbColor + c] = val;
         }
      }

      /* If we have another round of smoothing, gather the necassary data
       * and store those in a temporary array */
      for (int i = 0; i < nNodes; i++) {
         int sendCount = tmpRowSendCountsOfNode[i * nNodes + rank];
         int sendOffset = colRecvDisplOfNode[i * nNodes + rank];
         if (i == 0) {
            sendCount -= colsPerNode[rank] * _nbColor;
            sendOffset = colsPerNode[rank] * _nbColor;
         } else if (i == (nNodes - 1))
            sendCount -= colsPerNode[rank] * _nbColor;
         MPI_Igatherv(&tmpCol[sendOffset],
                      sendCount,
                      MPI_INT,
                      &tmpRow[0],
                      &tmpColRecvCountsOfNode[i * nNodes],
                      &tmpColSendDisplOfNode[i * nNodes],
                      MPI_INT,
                      i,
                      MPI_COMM_WORLD,
                      &req[i]);
      }
      MPI_Waitall(nNodes, &req[0], &stat[0]);

      /* Here we reorder the memory again but also do the normalization
       * for each pixel and store the result in the working array rowData */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (size_t irow = 0; irow < workNumRows; irow++) {
         int xOffset = 1;
         const int y = yOffset + irow;
         const int normPart = (min(y + range, _height - 1) - max(0, y - range) + 1);
         for (size_t inode = 0; inode < nNodes; inode++) {
            const int nodeRowLen = colsPerNode[inode] * _nbColor;
            const auto srcBegin = &tmpRow[tmpColSendDisplOfNode[rank * nNodes + inode]
                                          + irow * nodeRowLen];
            const auto destBegin = &rowData[rowOffset + rowLen * irow + _nbColor
                                            + tmpRowSendDisplOfNode[rank * nNodes + inode] / rowsPerNode[rank]];
            for (int j = 0; j < nodeRowLen; j++) {
               const int x = xOffset + j / _nbColor;
               const int norm = (min(x + range, _width - 1) - max(0, x - range) + 1) *
                                normPart - 1;
               destBegin[j] = (srcBegin[j] - destBegin[j]) / norm;
            }
            xOffset += colsPerNode[inode];
         }
      }

      const auto timeEnd = Clock::now();
      chrono::duration<double> elapsed = timeEnd - timeStart;
      ss.str("");
      ss << label << "[LOG] step " << iStep << ": done " << elapsed.count() << " s" << endl;
      cout << ss.str();
   }

   if (MPI_Gatherv(&rowData[0],
                   rowSendCounts[rank],
                   MPI_UNSIGNED_CHAR,
                   &_data[0],
                   &rowSendCounts[0],
                   &rowSendDispl[0],
                   MPI_UNSIGNED_CHAR,
                   0,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
      ss.str("");
      ss << label << "Final gathering failed";
      throw ss.str();
   }
}

#else
void PNGImage::smoothOMP(int nbSteps, int range)
{
   /* allocate temporary array, storing the smoothing
    * in the x-direction. We use integer as type in
    * order to avoid overflow of the png_byte. We also
    * use a continues memory layout, because this will
    * give us an advantage when using MPI */
   const size_t rowLen = _width * _nbColor;
   auto tmp = make_unique<int[]>(_height * rowLen);

   for (int iStep = 0; iStep < nbSteps; iStep++) {
      const auto timeStart = Clock::now();
      cout << "[LOG] step " << iStep << " started" << endl;

      /* smooth along the x-direction and store the results
       * in the temporary array. The algorithm starts, by
       * calculating the non-normalized value for the first
       * pixel and continues to calculate the values along
       * the x-direction by using the value of the previous
       * pixel. */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < _height * _nbColor; i++) {
         const int y = i / _nbColor;
         const int c = i % _nbColor;
         int val = 0;
         for (int x = 0; x <= range; x++)
            val += _data[y * rowLen + x * _nbColor + c];
         tmp[y * rowLen + c] = val;
         for (int x = 1; x < _width; x++) {
            const int l = x - range - 1;
            const int r = x + range;
            if (l >= 0)
               val -= _data[y * rowLen + l * _nbColor + c];
            if (r < _width)
               val += _data[y * rowLen + r * _nbColor + c];
            tmp[y * rowLen + x * _nbColor + c] = val;
         }
      }

      /* Finally smooth along the y-direction. The result is
       * stored directly in the _data array. This is thread-safe
       * because each threads writs in its unique memory space.
       * Further, the normilization is done here */
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < (_width - 2) * _nbColor; i++) {
         const int x = i / _nbColor + 1;
         const int c = i % _nbColor;
         int val = 0;
         for (int y = 0; y <= range; y++)
            val += tmp[y * rowLen + x * _nbColor + c];
         for (int y = 1; y < (_height - 1); y++) {
            const int t = y - range - 1;
            const int b = y + range;
            if (t >= 0)
               val -= tmp[t * rowLen + x * _nbColor + c];
            if (b < _height)
               val += tmp[b * rowLen + x * _nbColor + c];
            const int norm = (std::min(x + range, _width - 1) - std::max(0, x - range) + 1) *
                    (std::min(y + range, _height - 1) - std::max(0, y - range) + 1) - 1;
            _data[y * rowLen + x * _nbColor + c] = (val - _data[y * rowLen + x * _nbColor + c]) / norm;
         }
      }

      const auto timeEnd = Clock::now();
      chrono::duration<double> elapsed = timeEnd - timeStart;
      cout << "[LOG] step " << iStep << " done " << elapsed.count() << " s" << endl;
   }
}
#endif