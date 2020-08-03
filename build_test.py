import tensorrt as trt
import numpy as np

def custom_nms(inputs, network, custom_nms_plugin_creator):
    shareLocation = trt.PluginField("shareLocation", np.array([False], dtype=np.int32), trt.PluginFieldType.INT32)
    # varianceEncodedInTarget = trt.PluginField("varianceEncodedInTarget", np.array([True], dtype=np.int32), trt.PluginFieldType.INT32)
    backgroundLabelId = trt.PluginField("backgroundLabelId", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    numClasses = trt.PluginField("numClasses", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
    boxDims = trt.PluginField("boxDims", np.array([5], dtype=np.int32), trt.PluginFieldType.INT32)
    topK = trt.PluginField("topK", np.array([1024], dtype=np.int32), trt.PluginFieldType.INT32)
    keepTopK = trt.PluginField("keepTopK", np.array([50], dtype=np.int32), trt.PluginFieldType.INT32)
    scoreThreshold = trt.PluginField("scoreThreshold", np.array([0.3], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    iouThreshold = trt.PluginField("iouThreshold", np.array([0.9], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    # codeType = trt.PluginField("codeType", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    # inputOrder = trt.PluginField("inputOrder", np.array([0,1,2], dtype=np.int32), trt.PluginFieldType.INT32)
    # confSigmoid = trt.PluginField("confSigmoid", np.array([False], dtype=np.int32), trt.PluginFieldType.INT32)
    # isNormalized = trt.PluginField("isNormalized", np.array([True], dtype=np.int32), trt.PluginFieldType.INT32)

    pnms = trt.PluginFieldCollection([
        shareLocation,
        # varianceEncodedInTarget, 
        backgroundLabelId,
        numClasses,
        boxDims,
        topK,
        keepTopK,
        scoreThreshold,
        iouThreshold,
        # codeType,
        # inputOrder, 
        # confSigmoid, 
        # isNormalized
        ])

    custom_nms_plugin = custom_nms_plugin_creator.create_plugin('customnmsplugin', pnms)
    print('here')
    custom_nms_layer = network.add_plugin_v2([inp.get_output(0) for inp in inputs], custom_nms_plugin)
    print('here---')
    return custom_nms_layer, network

def custom_batchednms(inputs, network, batched_nms_plg_creator):
    shareLocation = trt.PluginField("shareLocation", np.array([False], dtype=np.int32), trt.PluginFieldType.INT32)
    # varianceEncodedInTarget = trt.PluginField("varianceEncodedInTarget", np.array([True], dtype=np.int32), trt.PluginFieldType.INT32)
    backgroundLabelId = trt.PluginField("backgroundLabelId", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    numClasses = trt.PluginField("numClasses", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
    topK = trt.PluginField("topK", np.array([1024], dtype=np.int32), trt.PluginFieldType.INT32)
    keepTopK = trt.PluginField("keepTopK", np.array([50], dtype=np.int32), trt.PluginFieldType.INT32)
    scoreThreshold = trt.PluginField("scoreThreshold", np.array([0.3], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    iouThreshold = trt.PluginField("iouThreshold", np.array([0.9], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    # codeType = trt.PluginField("codeType", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    # inputOrder = trt.PluginField("inputOrder", np.array([0,1,2], dtype=np.int32), trt.PluginFieldType.INT32)
    # confSigmoid = trt.PluginField("confSigmoid", np.array([False], dtype=np.int32), trt.PluginFieldType.INT32)
    # isNormalized = trt.PluginField("isNormalized", np.array([True], dtype=np.int32), trt.PluginFieldType.INT32)

    pnms = trt.PluginFieldCollection([
        shareLocation,
        # varianceEncodedInTarget, 
        backgroundLabelId,
        numClasses,
        topK,
        keepTopK,
        scoreThreshold,
        iouThreshold,
        # codeType,
        # inputOrder, 
        # confSigmoid, 
        # isNormalized
        ])

    batched_nms_plugin = batched_nms_plg_creator.create_plugin('batchednmsplugin', pnms)
    batched_nms_layer = network.add_plugin_v2([inp.get_output(0) for inp in inputs], batched_nms_plugin)
    return batched_nms_layer, network


if __name__ == "__main__":
	logger = trt.Logger(trt.Logger.WARNING)
	trt.init_libnvinfer_plugins(logger, '')
	plt_registry = trt.get_plugin_registry()

	print(len([a.name for a in plt_registry.plugin_creator_list]))
	print([a.name for a in plt_registry.plugin_creator_list])
	custom_nms_plugin_creator = plt_registry.get_plugin_creator("CustomNMS_TRT", "1", "")
	batched_nms_plugin_creator = plt_registry.get_plugin_creator("BatchedNMS_TRT", "1", "")

	with trt.Builder(logger) as builder, builder.create_network() as network:

		builder.max_batch_size = 1
		builder.max_workspace_size = 1 << 30

		bag = []
		np_box = np.array([
			[[0.1, 0.2, 0.1, 0.5, 1],
			[0.1, 0.3, 0.1, 0.6, 2],
			[0.3, 0.4, 0.6, 0.9, 3],],

			[[0.1, 0.2, 0.1, 0.5, 1],
			[0.1, 0.3, 0.1, 0.6, 2],
			[0.3, 0.4, 0.6, 0.9, 3],],

			[[0.1, 0.2, 0.1, 0.5, 1],
			[0.1, 0.3, 0.1, 0.6, 2],
			[0.3, 0.4, 0.6, 0.9, 3],],
			]).astype(np.float32).reshape(-1)

		np_conf = np.array([
			[0.1, 0.0, 0.9],
			[0.1, 0.2, 0.7],
			[0.1, 0.85, 0.05]]).astype(np.float32).reshape(-1)

		box = network.add_constant(trt.Dims([3, 3, 5]), 
			trt.Weights(np_box))
		box.name = 'box'

		conf = network.add_constant(trt.Dims([3, 3]), 
			trt.Weights(np_conf))
		conf.name = 'conf'

		# custom_nms, network = custom_nms([box, conf], network, custom_nms_plugin_creator)

		box_4dims = network.add_slice(box.get_output(0), trt.Dims([0,0]), trt.Dims([3,3,4]), trt.Dims([1,1,1]))
		print(box_4dims.shape)

		custom_nms, network = custom_batchednms([box_4dims, conf], network, batched_nms_plugin_creator)

		# topk_layer = network.add_topk(arange_tensor.get_output(0), 
		# 	trt.TopKOperation.MAX, 1000, 1)

		network.mark_output(custom_nms.get_output(0))
		engine_path = 'custom_nms.engine'
		with builder.build_cuda_engine(network) as engine:
			with open(engine_path, "wb") as f:
				f.write(engine.serialize())
			logging.info('Finish writing trt engine!')
			inputs, outputs, bindings, stream = allocate_buffers(engine)
			with engine.create_execution_context() as context:
				# case_num, inp = load_test_case(pagelocked_buffer=inputs[0].host, 
				#     height=args.ori_image_height, width=args.ori_image_width)
				# For more information on performing inference, refer to the introductory samples.
				# The common.do_inference function will return a list of outputs - we only have one in this case.
				output = do_inference(context, bindings=bindings, inputs=[], outputs=outputs, stream=stream)
				# 
				print(output)