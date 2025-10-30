FROM rocker/r-base:4.3.1

# Make a directory
RUN mkdir R-app

# Set working directory
WORKDIR /R-app

# Add dependencies file to R-app
COPY install_packages.R /R-app/

#Install packages
RUN Rscript install_packages.R

# Copy the R script
COPY main.R .

# Run the script
CMD ["Rscript", "main.R"]


# Download Kaggle datasets (put under src/data)
# Under titanic-prediction directory, build and run the Docker container:
# docker build -t titanic-pred-r  ./src/R-ver
# docker run --rm -v "$(pwd)/src/data:/R-app/data" titanic-pred-r