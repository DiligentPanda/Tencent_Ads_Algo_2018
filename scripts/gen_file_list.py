folder = "output/M_ff/test2"
output_fn = "output/M_ff/test2_file_list.txt"

import os
with open(output_fn,'w') as fw:
    for fn in os.listdir(folder):
        fw.write("output/M_ff/test2/"+fn+" 1"+"\n")

