import os

folder = '../bb/pitchjaw/invisible_filtered/'
for file in os.listdir(folder):
    # print(file)
    # # construct current name using file name and path
    old_name = os.path.join(folder, file)
    # # get file name without extension
    only_name = os.path.splitext(file)[0]
    # print(only_name[:-9])
    #
    # # Adding the new name with extension
    new_base = only_name[:-9] + '.csv'
    # # construct full file path
    new_name = os.path.join(folder, new_base)
    #
    # # Renaming the file
    os.rename(old_name, new_name)