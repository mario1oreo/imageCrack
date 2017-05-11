# http://cpquery.sipo.gov.cn/freeze.main?txn-code=createImgServlet&freshStept=1&now=2323123
# import sys
# from pyocr import pyocr
# from PIL import Image
#
# tools = pyocr.get_available_tools()[:]
# if len(tools) == 0:
#     print("No OCR tool found")
#     sys.exit(1)
# print("Using '%s'" % (tools[0].get_name()))
# image = Image.open('C:/image/pythonImage/1.bmp')
# code = tools[0].image_to_string(image)
# print ('code:', code)



# from PIL import Image
# import sys
# import pyocr
# import pyocr.builders
#
# image_path = 'C:/image/pythonImage/1.bmp'
# tools = pyocr.get_available_tools()
# if len(tools) == 0:
#     print("No OCR tool found")
#     sys.exit(1)
# tool = tools[0]
# print("Will use tool '%s'" % (tool.get_name()))
# # Ex: Will use tool 'tesseract'
# langs = tool.get_available_languages()
# print("Available languages: %s" % ", ".join(langs))
# lang = langs[1]
# print("Will use lang '%s'" % (lang))
# # Ex: Will use lang 'fra'
# txt = tool.image_to_string(Image.open(image_path))
# print txt

from PIL import Image
from pytesser import image_file_to_string as image_to_string

