import os
import time

import numpy as np
import pandas as pd

# AWS SDK
import boto3

# GCP SDK
from googleapiclient import discovery
from google.oauth2 import service_account

# Azure SDK
from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient

# Hyperparameters
ALPHA = 0.1       # learning rate
GAMMA = 0.9       # discount factor
EPSILON = 0.2     # exploration rate
