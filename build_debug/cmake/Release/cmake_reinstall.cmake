# Copyright (C) 2019 LAAS-CNRS, JRL AIST-CNRS, INRIA.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
if(EXISTS "/home/skleff/force_feedback_ws/force_feedback_mpc/build_debug/install_manifest.txt")
  execute_process(COMMAND "/home/skleff/miniconda3/envs/debug/bin/cmake" --build "/home/skleff/force_feedback_ws/force_feedback_mpc/build_debug" --target uninstall --config Release)
endif()
execute_process(COMMAND "/home/skleff/miniconda3/envs/debug/bin/cmake" --build "/home/skleff/force_feedback_ws/force_feedback_mpc/build_debug" --target install --config Release)
