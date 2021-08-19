import os

installed_module_list = os.popen("pip freeze")

# print(installed_module_list)
with open("C:/Users/Xizhi Huang/Desktop/requirements.txt", 'w') as f:
    for m in installed_module_list.read():
        f.write(m)