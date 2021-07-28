.PHONY: check clean

ultraface.onnx:
	wget -O $@ 'https://github.com/onnx/models/raw/master/vision/body_analysis/ultraface/models/version-RFB-640.onnx'

ultraface-post.onnx: ultraface.onnx
	onnigiri $^ -o $@ --from 460 --to boxes

ultraface_post.py: ultraface-post.onnx
	onnion $^ -o $@ --export-tensor-size 30

check: ultraface-post.onnx ultraface_post.py
	python ../check_model.py ultraface $^

clean:
	rm -rf *.onnx *.npy ultraface_post.py
