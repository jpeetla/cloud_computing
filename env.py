class CloudEnv:
    """
    Env wrapper to interact with AWS Lambda, GCP Cloud Functions, and Azure Functions.
    Each action = invoke one provider. State = real-time metrics from each.
    """
    def __init__(
        self,
        aws_func_name: str,
        gcp_project: str,
        gcp_func_name: str,
        azure_subscription_id: str,
        azure_resource_group: str,
        azure_func_app: str,
        alpha: float = ALPHA,
        beta: float = 0.5
    ):
        # AWS client
        self.aws = boto3.client("lambda",
                                 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                 aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                 region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.aws_func = aws_func_name

        # GCP client
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GCP_SA_JSON_PATH"))
        self.gcp = discovery.build("cloudfunctions", "v1", credentials=creds)
        self.gcp_name = f"projects/{gcp_project}/locations/us-central1/functions/{gcp_func_name}"

        # Azure client
        cred = DefaultAzureCredential()
        self.azure = WebSiteManagementClient(cred, azure_subscription_id)
        self.azure_rg = azure_resource_group
        self.azure_app = azure_func_app

        # reward weights
        self.alpha = alpha
        self.beta = beta

    def get_state(self) -> np.ndarray:
        """
        Fetch current metrics for each provider:
         - approximate energy (e.g. via CloudWatch / Stackdriver / Azure Monitor)
         - current avg latency
         - current load
        Return a state vector, e.g. [aws_energy, aws_latency, gcp_energy, …].
        """
        # TODO: call monitoring APIs to fill these lists
        aws_metrics = [0.0, 0.0]  
        gcp_metrics = [0.0, 0.0]
        az_metrics  = [0.0, 0.0]
        return np.array(aws_metrics + gcp_metrics + az_metrics)

    def step(self, action: int, payload: dict = None) -> dict:
        """
        Invoke the chosen cloud function:
          action == 0 → AWS
                 == 1 → GCP
                 == 2 → Azure
        Measure execution time, estimate energy & cost.
        Returns a dict with keys: 'latency', 'energy', 'cost'.
        """
        payload = payload or {"test": "data"}
        start = time.time()

        if action == 0:
            resp = self.aws.invoke(FunctionName=self.aws_func, Payload=json.dumps(payload))
            # parse resp if needed
        elif action == 1:
            resp = self.gcp.projects().locations().functions().call(
                name=self.gcp_name, body={"data": payload}
            ).execute()
        elif action == 2:
            # Azure Invocation via REST trigger URL (for example)
            url = self.azure_app.default_host_name + "/api/" + self.azure_app
            # req = requests.post(url, json=payload, headers={"x-functions-key": YOUR_KEY})
            # resp = req.json()
            pass
        else:
            raise ValueError("Invalid action")

        latency = time.time() - start
        # TODO: fetch actual energy & cost via cloud monitor APIs
        energy = 0.0
        cost = 0.0

        return {"latency": latency, "energy": energy, "cost": cost}

    def compute_reward(self, state: np.ndarray, action: int, metrics: dict) -> float:
        """
        Reward = –(energy + alpha*latency + beta*cost)
        We negate because lower energy/latency/cost is better.
        """
        return -(
            metrics["energy"] +
            self.alpha * metrics["latency"] +
            self.beta * metrics["cost"]
        )
