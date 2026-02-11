import jams
import os

# Đường dẫn đến file JAMS bất kỳ trong dataset của bạn
jams_path = "../../data/annotation/00_BN1-129-Eb_comp.jams" 

if os.path.exists(jams_path):
    jam = jams.load(jams_path)
    
    print("=== KIỂM TRA METADATA DÂY ĐÀN ===")
    found_strings = False
    
    for ann in jam.annotations:
        if ann.namespace == 'note_midi':
            # Đây là chỗ Leader không nhìn thấy:
            string_num = ann.annotation_metadata.data_source
            print(f"✅ Tìm thấy track dữ liệu cho: Dây số {string_num}")
            found_strings = True
            
            # In thử 1 nốt trong dây này
            if len(ann) > 0:
                note = ann.data[0]
                print(f"   -> Nốt đầu tiên: Pitch {note.value:.2f} (MIDI) chơi trên dây {string_num}")

    if found_strings:
        print("\n=> KẾT LUẬN: Dataset CÓ thông tin dây đàn (String).")
    else:
        print("\n=> KẾT LUẬN: Không tìm thấy.")
else:
    print("Không tìm thấy file JAMS để test.")