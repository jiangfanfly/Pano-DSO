# # TUM_MONO dataset
# add_executable( run_dso_tum_mono run_dso_tum_mono.cc )
# target_link_libraries( run_dso_tum_mono
#   ldso ${THIRD_PARTY_LIBS} )

# # EUROC dataset
# add_executable( run_dso_euroc run_dso_euroc.cc )
# target_link_libraries( run_dso_euroc
#   ldso ${THIRD_PARTY_LIBS} )

# # Kitti dataset
# add_executable( run_dso_kitti run_dso_kitti.cc )
# target_link_libraries( run_dso_kitti
#   ldso ${THIRD_PARTY_LIBS} )

# ladybug dataset
add_executable( run_panodso_multi_fisheye run_panodso_multi_fisheye.cc )
target_link_libraries( run_panodso_multi_fisheye
  ldso ${THIRD_PARTY_LIBS} )


# test some function
add_executable( testsomething testsomething.cc )
target_link_libraries( testsomething
  ldso ${THIRD_PARTY_LIBS} )
