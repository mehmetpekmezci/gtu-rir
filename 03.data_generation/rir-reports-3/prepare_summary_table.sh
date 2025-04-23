RIR_REPORT_DIR=~/RIR_REPORT_BACKUP/RIR_REPORT.EVAL.MORE.NODE.AND.MORE.HEAD

(
echo "DATASET;NODE;HEAD;MSE;SSIM;GLITCH;MESH_MSE"
for dataset in GTURIR BUTReverbDB
do
	for report_dir in $RIR_REPORT_DIR/$dataset/MESHTAE.node.*
	do
		node=$(basename $report_dir|sed -e 's/.*node.//'|cut -d. -f1)
		head=$(basename $report_dir|sed -e 's/.*node.//'|cut -d. -f3)
		mse_ssim_glitch=$(cat $report_dir/summary/*.txt | cut -d= -f2 | tr '\n' ';')
		mesh_reconst_loss_sum=$(cat $report_dir/METADATA_*/Mesh*/loss_records_for_mesh_reconstruction*.txt| cut -d= -f2 | tr '\n' '+')
		mesh_reconst_loss_num=$(wc -l $report_dir/METADATA_*/Mesh*/loss_records_for_mesh_reconstruction*.txt| awk '{print $1}'| head -1)
		mesh_reconst_loss_avg=$(echo "print((0${mesh_reconst_loss_sum}0)/$mesh_reconst_loss_num)" | python3)
		echo "$dataset;$node;$head;$mse_ssim_glitch$mesh_reconst_loss_avg"

        done
done
) > summary_table.csv
