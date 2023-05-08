System Requirement : python 3.8

To run the dressing-in-order pipeline, follow the steps below:
1. cd dressing-in-order/
2. pip install -r requirements.txt
3. python download_data.py
4. Replace standard_test_anns.txt in the data/ folder to run on multiple images
5. mkdir output - for pose-transfer.
6. mkdir output_tryon - for try on results.
7. mkdir output_pipeline - for pipeline results.
8. To run pose-transfer run "python dior_pose_transfer.py"
9. To run virtual try on, run "python tryon.py"
10. To run pipeline, run "python pipeline.py"


To run the pose-with-style, follow the steps below:
1. cd pose-with-style/
2. pip install -r requirements.txt
5. Run the Lets_try_and_pose.ipynb file



References:
The following repositories were used while developing this project: 
1. https://github.com/cuiaiyu/dressing-in-order
2. https://github.com/BadourAlBahar/pose-with-style