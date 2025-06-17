#! /bin/bash

# Define your actual server hostname
YOUR_ACTUAL_HOSTNAME="ec2-18-216-124-225.us-east-2.compute.amazonaws.com"
# Remove trailing / if it exists
YOUR_ACTUAL_HOSTNAME=${YOUR_ACTUAL_HOSTNAME%/}
# Use sed to replace placeholder in the HTML file
perl -pi -e "s|<your-server-hostname>|${YOUR_ACTUAL_HOSTNAME}|g" webarena-homepage/templates/index.html

cd webarena_homepage
flask run --host=0.0.0.0 --port=4399


