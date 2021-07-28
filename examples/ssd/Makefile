.PHONY: check clean

ssd-10.onnx:
	wget -O $@ https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd/model/ssd-10.onnx

ssd-10-post.onnx: ssd-10.onnx
	onnigiri $^ -o $@ --from Transpose_472 Transpose_661 --to bboxes labels scores

ssd_post.py: ssd-10-post.onnx
	onnion $^ -o $@ --export-tensor-size 30

check: ssd-10-post.onnx ssd_post.py
	python ../check_model.py ssd $^

clean:
	rm -rf ssd-10.onnx ssd-10-post.onnx *.npy ssd_post.py
