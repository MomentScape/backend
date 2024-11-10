FROM registry.hf.space/tencentarc-instantmesh:latest

# Install Flask
RUN pip install flask


# Copy the main application file
COPY run_img2threeD.py /home/user/app/

# Command to run the Flask application
CMD ["python", "run_img2threeD.py"]

# docker build -t img2threed .