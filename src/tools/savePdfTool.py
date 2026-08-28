from fpdf import FPDF

def saveText2Pdf(text: str, filename: str = "task_output.pdf") -> str:
	"""
	使用这个工具来保存任务输出为PDF文件，支持中文。
	:param text: 要保存的文本内容
	:param filename: 保存的PDF文件名
	:return: 保存状态消息
	"""
	# 初始化PDF
	pdf = FPDF()
	pdf.add_page()
	# 添加和设置字体，使用支持中文的字体文件（如 simhei.ttf）
	pdf.add_font('SimHei', '', 'simhei.ttf', uni=True) # 注册的字体，字体样式，字体文件，是否启用Unicode
	pdf.set_font('SimHei', '', 16)  # 设置当前字体，字体样式，字号
	# 添加标题
	pdf.cell(200, 10, '', ln=True, align='C')   # 创建的单元格宽高，内容，写完此行是否换行，居中对齐
	# 写入文本内容（支持中文）
	pdf.set_font('SimHei', '', 12)
	pdf.multi_cell(0, 10, text) # 输入多行文本，0：宽度占满当前剩余宽度，10：每行高度。
	# 保存PDF文件
	pdf.output("output/"+filename)

	return f"请前往 {filename} 查看报告"




