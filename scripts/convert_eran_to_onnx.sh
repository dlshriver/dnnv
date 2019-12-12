input_model_dir=${1%/}; shift
output_model_dir=${1%/}; shift

"mkdir -p $output_model_dir"
mkdir -p $output_model_dir

for input_path in ${input_model_dir}/*
do
    output_path="${output_model_dir}/$(basename ${input_path%.*}).onnx"
    echo "python tools/eran2onnx.py $input_path -o $output_path $@"
    python tools/eran2onnx.py $input_path -o $output_path $@
done