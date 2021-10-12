import os
import sys
import xml.dom.minidom


# encoding=utf-8
def parseTaskFile(taskFile):
    dom = xml.dom.minidom.parse(taskFile)

    root = dom.documentElement
    datas = root.getElementsByTagName('datas')[0]
    inputs = datas.getElementsByTagName('inputs')[0]
    inputList = inputs.getElementsByTagName('input')
    for input in inputList:
        print(input.getAttribute('name'), ':', input.firstChild.data)

    outputs = datas.getElementsByTagName('outputs')[0]
    outputList = outputs.getElementsByTagName('output')
    for output in outputList:
        print(output.getAttribute('name'), ':', output.firstChild.data)
        fp = open(output.firstChild.data, 'w')
        fp.close()

    parameters = root.getElementsByTagName('parameters')[0]
    parameterList = parameters.getElementsByTagName('parameter')
    for parameter in parameterList:
        print(parameter.getAttribute('name'), ':', parameter.firstChild.data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("argument error!")
        sys.exit(-1)
    taskFile = sys.argv[1]
    parseTaskFile(taskFile)
    #parseTaskFile("C:\\Users\\Xizhi Huang\\Desktop\\test.xml")
