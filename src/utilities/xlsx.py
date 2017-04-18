# Author: Cecil Wang (cecilwang@126.com)

import openpyxl


def extract_cells(filepath, start_cell, end_cell):
    values = []

    wb = openpyxl.load_workbook(filepath)
    ws = wb['Scores']
    cells = ws[start_cell: end_cell]
    for cell in cells:
        values.append(cell[0].value)

    return values


if __name__ == '__main__':
    extract_cells('D:/Project/Shakiness/data/scores.xlsx',
                  'V3', 'V502')
