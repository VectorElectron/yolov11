import onnx
from onnx import compose

def merge_yolo_pipeline_fixed(path1, path2, path3, output_path):
    m1 = onnx.load(path1)  # yolo_resize (Output: imgf, scale)
    m2 = onnx.load(path2)  # yolo11n     (Input: images; Output: grid)
    m3 = onnx.load(path3)  # yolo11_nms  (Input: grid, scale, ...; Output: boxes, scores...)

    m1 = compose.add_prefix(m1, prefix="res_")
    m2 = compose.add_prefix(m2, prefix="core_")
    m3 = compose.add_prefix(m3, prefix="nms_")

    m1_out_imgf = m1.graph.output[0].name   # 对应 imgf
    m1_out_scale = m1.graph.output[1].name  # 对应 scale
    
    m2_in_images = m2.graph.input[0].name   # 对应 images
    m2_out_grid = m2.graph.output[0].name   # 对应 grid
    
    m3_in_grid = next(i.name for i in m3.graph.input if "grid" in i.name)
    m3_in_scale = next(i.name for i in m3.graph.input if "scale" in i.name)

    print(f"数据流向 1: {m1_out_imgf} -> {m2_in_images}")
    print(f"数据流向 2 (跨层): {m1_out_scale} -> {m3_in_scale}")
    print(f"数据流向 3: {m2_out_grid} -> {m3_in_grid}")

    combined_12 = compose.merge_models(
        m1, m2,
        io_map=[(m1_out_imgf, m2_in_images)]
    )

    final_model = compose.merge_models(
        combined_12, m3,
        io_map=[
            (m2_out_grid, m3_in_grid),
            (m1_out_scale, m3_in_scale)
        ]
    )

    max_opset = max(opset.version for opset in final_model.opset_import)
    del final_model.opset_import[:]
    opset_info = final_model.opset_import.add()
    opset_info.domain = '' # default domain
    opset_info.version = max_opset

    # 7. 保存与验证
    onnx.checker.check_model(final_model)
    onnx.save(final_model, output_path)
    print(f"✅ 成功合并！最终输入包含: {[i.name for i in final_model.graph.input]}")

merge_yolo_pipeline_fixed('../model/yolo_resize.onnx',
    '../model/yolo11n.onnx', '../model/yolo_nms.onnx', '../model/yolo11n_one.onnx')
