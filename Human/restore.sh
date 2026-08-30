#!/bin/bash
# Undo the reorganisation: move every file back to its original location.
# Regenerated from MANIFEST.csv. Safe to re-run; already-restored files are
# skipped and missing sources are reported at the end.
cd "$(dirname "$0")/.." || exit 1
restored=0; skipped=0

if [ -f "_organized/references/2110.05836v2.pdf" ]; then
  mv -n "_organized/references/2110.05836v2.pdf" "2110.05836v2.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/references/301_Benchmarking_Progress_to_I.pdf" ]; then
  mv -n "_organized/references/301_Benchmarking_Progress_to_I.pdf" "301_Benchmarking_Progress_to_I.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/references/physical reasoning.pdf" ]; then
  mv -n "_organized/references/physical reasoning.pdf" "physical reasoning.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/time_extract.py" ]; then
  mkdir -p "Data analysis"
  mv -n "_organized/code/data_analysis/time_extract.py" "Data analysis/time_extract.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex/Untitled.Rmd" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/code/data_analysis/ex/Untitled.Rmd" "Data analysis/ex/Untitled.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/boxplot_by_group.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/boxplot_by_group.png" "Data analysis/ex/boxplot_by_group.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/boxplot_nature.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/boxplot_nature.pdf" "Data analysis/ex/boxplot_nature.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/boxplot_nature.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/boxplot_nature.png" "Data analysis/ex/boxplot_nature.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex/effect_sizes_addon.R" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/code/data_analysis/ex/effect_sizes_addon.R" "Data analysis/ex/effect_sizes_addon.R" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/formal_barchart.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/formal_barchart.pdf" "Data analysis/ex/formal_barchart.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/formal_barchart.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/formal_barchart.png" "Data analysis/ex/formal_barchart.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/performance_by_group.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/performance_by_group.png" "Data analysis/ex/performance_by_group.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/pilot_barchart.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/pilot_barchart.pdf" "Data analysis/ex/pilot_barchart.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/pilot_barchart.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/pilot_barchart.png" "Data analysis/ex/pilot_barchart.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_acc_v1_extremes_table.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_acc_v1_extremes_table.pdf" "Data analysis/ex/video_acc_v1_extremes_table.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_acc_v1_extremes_table.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_acc_v1_extremes_table.png" "Data analysis/ex/video_acc_v1_extremes_table.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_pc1_extremes_table.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_pc1_extremes_table.pdf" "Data analysis/ex/video_pc1_extremes_table.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_pc1_extremes_table.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_pc1_extremes_table.png" "Data analysis/ex/video_pc1_extremes_table.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_pc2_extremes_table.pdf" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_pc2_extremes_table.pdf" "Data analysis/ex/video_pc2_extremes_table.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex/video_pc2_extremes_table.png" ]; then
  mkdir -p "Data analysis/ex"
  mv -n "_organized/figures/data_analysis/ex/video_pc2_extremes_table.png" "Data analysis/ex/video_pc2_extremes_table.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_cat/ex_cat.Rmd" ]; then
  mkdir -p "Data analysis/ex_cat"
  mv -n "_organized/code/data_analysis/ex_cat/ex_cat.Rmd" "Data analysis/ex_cat/ex_cat.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/Rplots.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/Rplots.pdf" "Data analysis/ex_cat_matrix/Rplots.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_variance.png" "Data analysis/ex_cat_matrix/cat_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" "Data analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.png" "Data analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" "Data analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" "Data analysis/ex_cat_matrix/cat_RT_PCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_video_RSM.pdf" "Data analysis/ex_cat_matrix/cat_RT_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_RT_video_RSM.png" "Data analysis/ex_cat_matrix/cat_RT_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" "Data analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.png" "Data analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" "Data analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" "Data analysis/ex_cat_matrix/cat_correctness_logisticPCA_video_scores_PC1_PC2_PC3_semantic_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_video_RSM.pdf" "Data analysis/ex_cat_matrix/cat_correctness_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/cat_correctness_video_RSM.png" "Data analysis/ex_cat_matrix/cat_correctness_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/correctness_logisticPCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/correctness_logisticPCA_scree.pdf" "Data analysis/ex_cat_matrix/correctness_logisticPCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/correctness_logisticPCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/correctness_logisticPCA_scree.png" "Data analysis/ex_cat_matrix/correctness_logisticPCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.pdf" "Data analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/figures/data_analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.png" "Data analysis/ex_cat_matrix/ex_cat_RT_PCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_cat_matrix/ex_cat_matrix.Rmd" ]; then
  mkdir -p "Data analysis/ex_cat_matrix"
  mv -n "_organized/code/data_analysis/ex_cat_matrix/ex_cat_matrix.Rmd" "Data analysis/ex_cat_matrix/ex_cat_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_sensor/ex_sensor.Rmd" ]; then
  mkdir -p "Data analysis/ex_sensor"
  mv -n "_organized/code/data_analysis/ex_sensor/ex_sensor.Rmd" "Data analysis/ex_sensor/ex_sensor.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/Rplots.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/Rplots.pdf" "Data analysis/ex_sensor_matrix/Rplots.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/correctness_logisticPCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/correctness_logisticPCA_scree.pdf" "Data analysis/ex_sensor_matrix/correctness_logisticPCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/correctness_logisticPCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/correctness_logisticPCA_scree.png" "Data analysis/ex_sensor_matrix/correctness_logisticPCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.pdf" "Data analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.png" "Data analysis/ex_sensor_matrix/ex_cat_RT_PCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_sensor_matrix/ex_sensor_matrix.Rmd" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/code/data_analysis/ex_sensor_matrix/ex_sensor_matrix.Rmd" "Data analysis/ex_sensor_matrix/ex_sensor_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.pdf" "Data analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.png" "Data analysis/ex_sensor_matrix/sensor_RT_PCA_pc123_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_PCA_variance.png" "Data analysis/ex_sensor_matrix/sensor_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_video_RSM.pdf" "Data analysis/ex_sensor_matrix/sensor_RT_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_RT_video_RSM.png" "Data analysis/ex_sensor_matrix/sensor_RT_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.pdf" "Data analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.png" "Data analysis/ex_sensor_matrix/sensor_correctness_logisticPCA_pc123_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_video_RSM.pdf" "Data analysis/ex_sensor_matrix/sensor_correctness_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_sensor_matrix"
  mv -n "_organized/figures/data_analysis/ex_sensor_matrix/sensor_correctness_video_RSM.png" "Data analysis/ex_sensor_matrix/sensor_correctness_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_voe/ex_voe.Rmd" ]; then
  mkdir -p "Data analysis/ex_voe"
  mv -n "_organized/code/data_analysis/ex_voe/ex_voe.Rmd" "Data analysis/ex_voe/ex_voe.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/Rplots.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/Rplots.pdf" "Data analysis/ex_voe_matrix/Rplots.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/correctness_logisticPCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/correctness_logisticPCA_scree.pdf" "Data analysis/ex_voe_matrix/correctness_logisticPCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/correctness_logisticPCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/correctness_logisticPCA_scree.png" "Data analysis/ex_voe_matrix/correctness_logisticPCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.pdf" "Data analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.png" "Data analysis/ex_voe_matrix/ex_cat_RT_PCA_scree.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/ex_voe_matrix/ex_voe_matrix.Rmd" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/code/data_analysis/ex_voe_matrix/ex_voe_matrix.Rmd" "Data analysis/ex_voe_matrix/ex_voe_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.pdf" "Data analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.png" "Data analysis/ex_voe_matrix/voe_RT_PC123_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.pdf" "Data analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.png" "Data analysis/ex_voe_matrix/voe_RT_PC123_semantic_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_PCA_variance.png" "Data analysis/ex_voe_matrix/voe_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM.pdf" "Data analysis/ex_voe_matrix/voe_RT_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM.png" "Data analysis/ex_voe_matrix/voe_RT_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.pdf" "Data analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.png" "Data analysis/ex_voe_matrix/voe_RT_video_RSM_similarity_clustered.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.pdf" "Data analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.png" "Data analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_flat_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.pdf" "Data analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.png" "Data analysis/ex_voe_matrix/voe_correctness_logisticPCA_PC123_semantic_biplot.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM.pdf" "Data analysis/ex_voe_matrix/voe_correctness_video_RSM.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM.png" "Data analysis/ex_voe_matrix/voe_correctness_video_RSM.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.pdf" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.pdf" "Data analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.png" "Data analysis/ex_voe_matrix/voe_correctness_video_RSM_similarity_clustered.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-14-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-14-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-14-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-16-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-16-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-16-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-17-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-17-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-17-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-19-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-19-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-19-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-20-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-20-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-20-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-21-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-21-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-21-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-23-1.png" ]; then
  mkdir -p "Data analysis/ex_voe_matrix/figure"
  mv -n "_organized/figures/data_analysis/ex_voe_matrix/unnamed-chunk-23-1.png" "Data analysis/ex_voe_matrix/figure/unnamed-chunk-23-1.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot/Rplots.pdf" ]; then
  mkdir -p "Data analysis/pilot"
  mv -n "_organized/figures/data_analysis/pilot/Rplots.pdf" "Data analysis/pilot/Rplots.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot/ancova.Rmd" ]; then
  mkdir -p "Data analysis/pilot"
  mv -n "_organized/code/data_analysis/pilot/ancova.Rmd" "Data analysis/pilot/ancova.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot/boxplot_by_group.png" ]; then
  mkdir -p "Data analysis/pilot"
  mv -n "_organized/figures/data_analysis/pilot/boxplot_by_group.png" "Data analysis/pilot/boxplot_by_group.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot/pilot_barchart.pdf" ]; then
  mkdir -p "Data analysis/pilot"
  mv -n "_organized/figures/data_analysis/pilot/pilot_barchart.pdf" "Data analysis/pilot/pilot_barchart.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot/pilot_barchart.png" ]; then
  mkdir -p "Data analysis/pilot"
  mv -n "_organized/figures/data_analysis/pilot/pilot_barchart.png" "Data analysis/pilot/pilot_barchart.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_1_cat/pilot1_cat.Rmd" ]; then
  mkdir -p "Data analysis/pilot_1_cat"
  mv -n "_organized/code/data_analysis/pilot_1_cat/pilot1_cat.Rmd" "Data analysis/pilot_1_cat/pilot1_cat.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot_cat_matrix/cat_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/pilot_cat_matrix"
  mv -n "_organized/figures/data_analysis/pilot_cat_matrix/cat_RT_PCA_variance.png" "Data analysis/pilot_cat_matrix/cat_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_cat_matrix/pilot_cat_matrix.Rmd" ]; then
  mkdir -p "Data analysis/pilot_cat_matrix"
  mv -n "_organized/code/data_analysis/pilot_cat_matrix/pilot_cat_matrix.Rmd" "Data analysis/pilot_cat_matrix/pilot_cat_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_demo/Untitled.Rmd" ]; then
  mkdir -p "Data analysis/pilot_demo"
  mv -n "_organized/code/data_analysis/pilot_demo/Untitled.Rmd" "Data analysis/pilot_demo/Untitled.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_sensor/sensor.Rmd" ]; then
  mkdir -p "Data analysis/pilot_sensor"
  mv -n "_organized/code/data_analysis/pilot_sensor/sensor.Rmd" "Data analysis/pilot_sensor/sensor.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_sensor_matrix/pilot_sensor_matrix.Rmd" ]; then
  mkdir -p "Data analysis/pilot_sensor_matrix"
  mv -n "_organized/code/data_analysis/pilot_sensor_matrix/pilot_sensor_matrix.Rmd" "Data analysis/pilot_sensor_matrix/pilot_sensor_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot_sensor_matrix/sensor_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/pilot_sensor_matrix"
  mv -n "_organized/figures/data_analysis/pilot_sensor_matrix/sensor_RT_PCA_variance.png" "Data analysis/pilot_sensor_matrix/sensor_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_voe/voe.Rmd" ]; then
  mkdir -p "Data analysis/pilot_voe"
  mv -n "_organized/code/data_analysis/pilot_voe/voe.Rmd" "Data analysis/pilot_voe/voe.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/data_analysis/pilot_voe_matrix/pilot_voe_matrix.Rmd" ]; then
  mkdir -p "Data analysis/pilot_voe_matrix"
  mv -n "_organized/code/data_analysis/pilot_voe_matrix/pilot_voe_matrix.Rmd" "Data analysis/pilot_voe_matrix/pilot_voe_matrix.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/data_analysis/pilot_voe_matrix/voe_RT_PCA_variance.png" ]; then
  mkdir -p "Data analysis/pilot_voe_matrix"
  mv -n "_organized/figures/data_analysis/pilot_voe_matrix/voe_RT_PCA_variance.png" "Data analysis/pilot_voe_matrix/voe_RT_PCA_variance.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/ex_lmm/lmm.Rmd" ]; then
  mkdir -p "ex_lmm"
  mv -n "_organized/code/ex_lmm/lmm.Rmd" "ex_lmm/lmm.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/build_rt_critical_workbook.mjs" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/build_rt_critical_workbook.mjs" "hddm_ex/build_rt_critical_workbook.mjs" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/export_rt_critical_data_package.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/export_rt_critical_data_package.py" "hddm_ex/export_rt_critical_data_package.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/hddm_dual_rt_analysis.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/hddm_dual_rt_analysis.py" "hddm_ex/hddm_dual_rt_analysis.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_ddm_schematic.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_ddm_schematic.py" "hddm_ex/make_ddm_schematic.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_hddm_example_style_figure.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_hddm_example_style_figure.py" "hddm_ex/make_hddm_example_style_figure.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_hddm_feature_modulation_figures.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_hddm_feature_modulation_figures.py" "hddm_ex/make_hddm_feature_modulation_figures.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_hddm_posterior_av_figure.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_hddm_posterior_av_figure.py" "hddm_ex/make_hddm_posterior_av_figure.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_hddm_result_figures.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_hddm_result_figures.py" "hddm_ex/make_hddm_result_figures.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_posterior_coefficients_figure_nature.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_posterior_coefficients_figure_nature.py" "hddm_ex/make_posterior_coefficients_figure_nature.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_rt_condition_sample_style_figures.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_rt_condition_sample_style_figures.py" "hddm_ex/make_rt_condition_sample_style_figures.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/make_visual_z_behavior_figure_nature.py" ]; then
  mkdir -p "hddm_ex"
  mv -n "_organized/code/hddm_ex/make_visual_z_behavior_figure_nature.py" "hddm_ex/make_visual_z_behavior_figure_nature.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_1_behavioral_summary.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_1_behavioral_summary.png" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_1_behavioral_summary.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_1_behavioral_summary.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_1_behavioral_summary.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_1_behavioral_summary.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_2_model_comparison.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_2_model_comparison.png" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_2_model_comparison.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_2_model_comparison.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_2_model_comparison.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_2_model_comparison.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_3_regression_parameter_summary.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_3_regression_parameter_summary.png" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_3_regression_parameter_summary.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_3_regression_parameter_summary.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_3_regression_parameter_summary.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_3_regression_parameter_summary.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_4_group_model_summary.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_4_group_model_summary.png" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_4_group_model_summary.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_4_group_model_summary.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_4_group_model_summary.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_4_group_model_summary.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_5_example_style_rt_critical.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_5_example_style_rt_critical.png" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_5_example_style_rt_critical.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_5_example_style_rt_critical.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/figure_5_example_style_rt_critical.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/figure_5_example_style_rt_critical.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_0_ddm_schematic.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_0_ddm_schematic.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_0_ddm_schematic.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_0_ddm_schematic.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_0_ddm_schematic.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_0_ddm_schematic.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_modulation.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_modulation.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_modulation_nature.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_modulation_nature.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_modulation_nature.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_modulation_nature.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_posterior_coefficients.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_posterior_coefficients.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_posterior_coefficients_nature.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_posterior_coefficients_nature.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_physical_z_posterior_coefficients_nature.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_physical_z_posterior_coefficients_nature.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_a_v_comparison.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_a_v_comparison.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_posterior_a_v_comparison.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_a_v_comparison.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_a_v_comparison.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_posterior_a_v_comparison.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_v_only.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_v_only.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_posterior_v_only.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_v_only.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_posterior_v_only.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_posterior_v_only.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_nature.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_nature.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_nature.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_nature.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_critical.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_critical.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_rt_critical.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_critical.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_critical.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_rt_critical.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_onset.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_onset.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_rt_onset.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_onset.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_modulation_rt_onset.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_modulation_rt_onset.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_posterior_coefficients.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_posterior_coefficients.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_posterior_coefficients_nature.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.png" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_posterior_coefficients_nature.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/paper_figure_visual_z_posterior_coefficients_nature.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/paper_figure_visual_z_posterior_coefficients_nature.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/figure_hddm_composite.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/figure_hddm_composite.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/figure_hddm_composite.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/figure_hddm_composite.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/figure_hddm_composite.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/figure_hddm_composite.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_a_ddm_traces.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_a_ddm_traces.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_a_ddm_traces.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_a_ddm_traces.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_a_ddm_traces.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_a_ddm_traces.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_b_boundary_ndt.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_b_boundary_ndt.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_b_boundary_ndt.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_b_boundary_ndt.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_b_boundary_ndt.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_b_boundary_ndt.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_c_visual_z_modulation.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_c_visual_z_modulation.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_c_visual_z_modulation.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_c_visual_z_modulation.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_c_visual_z_modulation.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_c_visual_z_modulation.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_d_visual_z_posterior_coefs.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_d_visual_z_posterior_coefs.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_d_visual_z_posterior_coefs.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_d_visual_z_posterior_coefs.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_d_visual_z_posterior_coefs.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_d_visual_z_posterior_coefs.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_e_physical_z_modulation.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_e_physical_z_modulation.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_e_physical_z_modulation.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_e_physical_z_modulation.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_e_physical_z_modulation.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_e_physical_z_modulation.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_f_physical_z_posterior_coefs.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_f_physical_z_posterior_coefs.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_f_physical_z_posterior_coefs.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_f_physical_z_posterior_coefs.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/panel_f_physical_z_posterior_coefs.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/panel_f_physical_z_posterior_coefs.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_a_ddm_traces_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_a_ddm_traces_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_a_ddm_traces_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_a_ddm_traces_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_a_ddm_traces_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_a_ddm_traces_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_b_boundary_ndt_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_b_boundary_ndt_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_b_boundary_ndt_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_b_boundary_ndt_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_b_boundary_ndt_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_b_boundary_ndt_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_c_visual_z_modulation_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_c_visual_z_modulation_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_c_visual_z_modulation_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_c_visual_z_modulation_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_c_visual_z_modulation_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_c_visual_z_modulation_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_d_visual_z_posterior_coefs_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_e_physical_z_modulation_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_e_physical_z_modulation_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_e_physical_z_modulation_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_e_physical_z_modulation_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_e_physical_z_modulation_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_e_physical_z_modulation_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.pdf" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.pdf" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.pdf" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.png" "hddm_ex/hddm_results_4chains_2000samples/figures/nature/no_key/panel_f_physical_z_posterior_coefs_nokey.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_action_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_action_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_action_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_action_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_action_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_action_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_intuitive_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_intuitive_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_intuitive_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_intuitive_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_intuitive_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_intuitive_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_semantic_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_semantic_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_semantic_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_semantic_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_critical_semantic_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_critical_semantic_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_action_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_action_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_action_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_action_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_action_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_action_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_intuitive_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_intuitive_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_intuitive_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_intuitive_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_intuitive_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_intuitive_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_semantic_sample_style.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_semantic_sample_style.png" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_semantic_sample_style.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_semantic_sample_style.svg" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_condition_sample_style/rt_onset_semantic_sample_style.svg" "hddm_ex/hddm_results_4chains_2000samples/figures/rt_condition_sample_style/rt_onset_semantic_sample_style.svg" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/group"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/group"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/group/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/null"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/null"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/null/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_critical/regression/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/group"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/group"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/group/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/null"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/null"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/null/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/posterior_predictive_rt_hist.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/posterior_predictive_rt_hist.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/posterior_predictive_rt_hist.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/trace_plots.png" ]; then
  mkdir -p "hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression"
  mv -n "_organized/figures/hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/trace_plots.png" "hddm_ex/hddm_results_4chains_2000samples/rt_onset/regression/trace_plots.png" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/manuscript_common/figure_style.py" ]; then
  mkdir -p "hddm_ex/manuscript_common"
  mv -n "_organized/code/hddm_ex/manuscript_common/figure_style.py" "hddm_ex/manuscript_common/figure_style.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/export_panels_nokey.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/export_panels_nokey.py" "hddm_ex/nature_panels/export_panels_nokey.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/figure_hddm_composite.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/figure_hddm_composite.py" "hddm_ex/nature_panels/figure_hddm_composite.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/generate_figure_1_caption.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/generate_figure_1_caption.py" "hddm_ex/nature_panels/generate_figure_1_caption.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/panel_a_ddm_traces.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/panel_a_ddm_traces.py" "hddm_ex/nature_panels/panel_a_ddm_traces.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/panel_b_boundary_ndt.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/panel_b_boundary_ndt.py" "hddm_ex/nature_panels/panel_b_boundary_ndt.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/panel_ce_feature_modulation.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/panel_ce_feature_modulation.py" "hddm_ex/nature_panels/panel_ce_feature_modulation.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_ex/nature_panels/panel_df_posterior_coefs.py" ]; then
  mkdir -p "hddm_ex/nature_panels"
  mv -n "_organized/code/hddm_ex/nature_panels/panel_df_posterior_coefs.py" "hddm_ex/nature_panels/panel_df_posterior_coefs.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/academic_figure_style.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/academic_figure_style.py" "hddm_project/scripts/academic_figure_style.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/draw_embodiment_spectrum_diagram.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/draw_embodiment_spectrum_diagram.py" "hddm_project/scripts/draw_embodiment_spectrum_diagram.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/generate_academic_figures.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/generate_academic_figures.py" "hddm_project/scripts/generate_academic_figures.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/make_hddm_results_table.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/make_hddm_results_table.py" "hddm_project/scripts/make_hddm_results_table.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_group_posteriors.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_group_posteriors.py" "hddm_project/scripts/plot_group_posteriors.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_hddm_rtcritical_group_posteriors_combined.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_hddm_rtcritical_group_posteriors_combined.py" "hddm_project/scripts/plot_hddm_rtcritical_group_posteriors_combined.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_hddm_taskwise_result_figures.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_hddm_taskwise_result_figures.py" "hddm_project/scripts/plot_hddm_taskwise_result_figures.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_hddm_taskwise_sets.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_hddm_taskwise_sets.py" "hddm_project/scripts/plot_hddm_taskwise_sets.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_physical_z_posterior_distributions.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_physical_z_posterior_distributions.py" "hddm_project/scripts/plot_physical_z_posterior_distributions.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_physical_z_reversal.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_physical_z_reversal.py" "hddm_project/scripts/plot_physical_z_reversal.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_rt_condition_sample_style.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_rt_condition_sample_style.py" "hddm_project/scripts/plot_rt_condition_sample_style.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_rt_hddm_taskwise_sets.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_rt_hddm_taskwise_sets.py" "hddm_project/scripts/plot_rt_hddm_taskwise_sets.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_visual_z_double_dissociation.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_visual_z_double_dissociation.py" "hddm_project/scripts/plot_visual_z_double_dissociation.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/plot_visual_z_posterior_distributions.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/plot_visual_z_posterior_distributions.py" "hddm_project/scripts/plot_visual_z_posterior_distributions.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/postprocess_hddm_results.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/postprocess_hddm_results.py" "hddm_project/scripts/postprocess_hddm_results.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/prepare_hddm_data.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/prepare_hddm_data.py" "hddm_project/scripts/prepare_hddm_data.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/run_hddm_models.py" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/run_hddm_models.py" "hddm_project/scripts/run_hddm_models.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/hddm_project/scripts/setup_hddm_apple_silicon.sh" ]; then
  mkdir -p "hddm_project/scripts"
  mv -n "_organized/code/hddm_project/scripts/setup_hddm_apple_silicon.sh" "hddm_project/scripts/setup_hddm_apple_silicon.sh" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/lmm/scripts/plot_physical_z_rt_effects.py" ]; then
  mkdir -p "lmm/scripts"
  mv -n "_organized/code/lmm/scripts/plot_physical_z_rt_effects.py" "lmm/scripts/plot_physical_z_rt_effects.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/lmm/scripts/plot_visual_z_rt_effects.py" ]; then
  mkdir -p "lmm/scripts"
  mv -n "_organized/code/lmm/scripts/plot_visual_z_rt_effects.py" "lmm/scripts/plot_visual_z_rt_effects.py" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project1_ex/cat/ex_cat.Rmd" ]; then
  mkdir -p "project1_ex/cat"
  mv -n "_organized/code/project1_ex/cat/ex_cat.Rmd" "project1_ex/cat/ex_cat.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project1_ex/sensor/ex_1_sensor.Rmd" ]; then
  mkdir -p "project1_ex/sensor"
  mv -n "_organized/code/project1_ex/sensor/ex_1_sensor.Rmd" "project1_ex/sensor/ex_1_sensor.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project1_ex/voe/ex_1_voe.Rmd" ]; then
  mkdir -p "project1_ex/voe"
  mv -n "_organized/code/project1_ex/voe/ex_1_voe.Rmd" "project1_ex/voe/ex_1_voe.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project2_ex/cat/ex_2_cat.Rmd" ]; then
  mkdir -p "project2_ex/cat"
  mv -n "_organized/code/project2_ex/cat/ex_2_cat.Rmd" "project2_ex/cat/ex_2_cat.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project2_ex/sensor/ex_2_sensor.Rmd" ]; then
  mkdir -p "project2_ex/sensor"
  mv -n "_organized/code/project2_ex/sensor/ex_2_sensor.Rmd" "project2_ex/sensor/ex_2_sensor.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi
if [ -f "_organized/code/project2_ex/voe/ex_2_voe.Rmd" ]; then
  mkdir -p "project2_ex/voe"
  mv -n "_organized/code/project2_ex/voe/ex_2_voe.Rmd" "project2_ex/voe/ex_2_voe.Rmd" && restored=$((restored+1))
else
  skipped=$((skipped+1))
fi

echo "restored=$restored skipped=$skipped total=444"
echo "Note: the path repairs inside the scripts are NOT reverted."
