"""Create synthetic demo fixtures. No course video, label or image is reused."""
from pathlib import Path
import cv2
import numpy as np
import openpyxl


def main():
    root = Path(__file__).resolve().parent / 'synthetic_demo'
    videos = root / 'videos'
    videos.mkdir(parents=True,exist_ok=True)
    width, height = 384,192
    for index in (0,1):
        writer = cv2.VideoWriter(str(videos / f'{index:02d}.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'),4,(width,height))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoding is unavailable in this OpenCV build')
        for t in range(8):
            top = np.zeros((64,width,3),dtype=np.uint8)
            box = np.zeros_like(top)
            for view in range(6):
                offset = view*64
                cv2.line(top,(offset+15,63),(offset+25,20),(200,200,200),2)
                cv2.line(top,(offset+49,63),(offset+39,20),(200,200,200),2)
                cv2.rectangle(top,(offset+25+t,27),(offset+37+t,42),(80,180,220),-1)
                cv2.rectangle(box,(offset+25+t,27),(offset+37+t,42),(0,220,0),1)
            gen = top.copy() if index == 0 else np.full_like(top,255 if t%2 else 0)
            writer.write(np.vstack([top,box,gen]))
        writer.release()
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(['SYNTHETIC DEMO: geometric fixtures, not real driving labels'])
    sheet.append(['Video index',None,None,None,'Semantic',None,None,None,
                  'Logical',None,None,None,'Decision',None,'Attack level',None,'Reasoning'])
    for index, score, level in ((0,0.0,'None'),(1,1.0,'Decision')):
        row = [None]*17
        row[0] = index
        for col in (4,8,12):
            row[col] = score
        row[14] = level
        row[16] = 'Synthetic demo fixture label; not a human judgment of driving safety.'
        sheet.append(row)
    workbook.save(root/'annotations.xlsx')
    print('Created synthetic demo inputs in examples/synthetic_demo')


if __name__ == '__main__':
    main()
