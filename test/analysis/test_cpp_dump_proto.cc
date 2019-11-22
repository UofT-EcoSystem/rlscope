//
// Created by jagle on 11/20/2019.
//

#include <limits.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "analysis/trace_file_parser.h"

using namespace Eigen;

namespace tensorflow {

//TEST(FailingTest, OneEqZero) {
// EXPECT_EQ(1, 0);
//}

TEST(TestCategoryKey, MapContainsKey) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);
}

TEST(TestCategoryKey, MapContainsKeyCopy) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);


  auto process_copy = "process";
  auto category_copy = "category";
  auto category_key_copy = CategoryKey::FromCategory(process_copy, category_copy);
  contains = (cmap.find(category_key_copy) != cmap.end());
  EXPECT_EQ(contains, true);
}

TEST(TestEigen, ElementWiseLessThan) {
  using IdxArray = Array<size_t, Dynamic, 1>;
  size_t k = 3;
  auto A = IdxArray::Constant(k, 1);
  auto B = IdxArray::Constant(k, 2);
  bool A_less_than_B = (A < B).any();
  EXPECT_TRUE(A_less_than_B);
}

TEST(TestEigen, ArrayModifyOne) {
//  using IdxArray = Array<size_t, Dynamic, 1>;
//  size_t k = 3;
//  auto A = IdxArray::Constant(k, 0);
//  IdxArray B;
//  B << 0, 1, 0;
//  // A(1) += 1;
//  A[1][0] = 1;
//  bool A_eq_B = (A == B).all();
//  EXPECT_TRUE(A_eq_B);

//  // using ArrayType = ArrayXXf;
//  using ArrayType = Array<size_t, Dynamic, Dynamic>;
//
//  ArrayType  m(2,2);
//
//  // assign some values coefficient by coefficient
//  m(0,0) = 1.0; m(0,1) = 2.0;
//  m(1,0) = 3.0; m(1,1) = m(0,1) + m(1,0);
//
//  ArrayType  m_comma(2,2);
//    // using the comma-initializer is also allowed
//  m_comma <<
//      1.0, 2.0,
//      3.0, 5.0;
//  EXPECT_TRUE((m == m_comma).all());


//  // using ArrayType = ArrayXXf;
//  using ArrayType = Array<size_t, Dynamic, 1>;
//
//  ArrayType  m(2,1);
//
//  // assign some values coefficient by coefficient
//  m(0,0) = 1.0;
//  m(1,0) = 3.0;
//
//  ArrayType  m_comma(2,1);
//  // using the comma-initializer is also allowed
//  m_comma <<
//      1.0,
//      3.0;
//  EXPECT_TRUE((m == m_comma).all());


  // using ArrayType = ArrayXXf;
  using ArrayType = Array<size_t, Dynamic, 1>;

  ArrayType  m(2);

  // assign some values coefficient by coefficient
  m(0) = 1.0;
  m(1) = 3.0;
  m(1) += 3.0;

  ArrayType  m_comma(2);
  // using the comma-initializer is also allowed
  m_comma <<
      1.0,
      6.0;
  EXPECT_TRUE((m == m_comma).all());

}

}
