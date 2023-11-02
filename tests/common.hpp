///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

// This file is a near direct copy of crocoddyl/unittest/unittest_common.hpp

/**
 * To be included last in the test_XXX.cpp,
 * otherwise it interferes with pinocchio boost::variant.
 */

#ifndef FORCE_FEEDBACK_MPC_UNITTEST_COMMON_HPP_
#define FORCE_FEEDBACK_MPC_UNITTEST_COMMON_HPP_

#define NUMDIFF_MODIFIER 10.

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <boost/function.hpp>
#include <boost/test/execution_monitor.hpp>  // for execution_exception
#include <boost/test/included/unit_test.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <iterator>
#include <string>

#endif  // FORCE_FEEDBACK_MPC_UNITTEST_COMMON_HPP_
