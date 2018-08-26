//
// Created by klaus on 25.08.18.
//

#ifndef NUMECA_CPP14_H
#define NUMECA_CPP14_H

#include <memory>

#if __cplusplus < 201400L
namespace std
{
// note: this implementation does not disable this overload for array types
template<class T, class ...Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&& ...args)
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
template<class T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(std::size_t n)
{
   typedef typename std::remove_extent<T>::type RT;
   return std::unique_ptr<T>(new RT[n]);
}
};
#endif

#endif //NUMECA_CPP14_H
