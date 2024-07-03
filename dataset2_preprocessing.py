import os

def modify_first_number_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            modify_first_number(file_path)

def modify_first_number(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        values = line.split()
        if values:
            first_number = int(values[0])
            modified_first_number = first_number + 100
            modified_line = f"{modified_first_number} {' '.join(values[1:])}"
            modified_lines.append(modified_line)

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

folder_path = 'test.txt'  
modify_first_number_in_folder(folder_path)