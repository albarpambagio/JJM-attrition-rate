# Setting up Metabase with Rancher Desktop

This guide explains how to set up Metabase with Rancher Desktop to visualize your data from `feature_monitor.db`.

## Prerequisites
- Rancher Desktop installed and running
- Kubernetes enabled in Rancher Desktop
- Basic knowledge of Kubernetes and kubectl commands

## Step 1: Create the Deployment Configuration

Create a file named `metabase-deployment.yaml` with the following configuration:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: feature-monitor-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: metabase-db-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: metabase
spec:
  selector:
    matchLabels:
      app: metabase
  template:
    metadata:
      labels:
        app: metabase
    spec:
      containers:
      - name: metabase
        image: metabase/metabase:latest
        env:
        - name: MB_DB_TYPE
          value: h2
        - name: MB_DB_FILE
          value: /metabase-data/metabase.db
        volumeMounts:
        - name: feature-monitor-storage
          mountPath: /data
        - name: metabase-db-storage
          mountPath: /metabase-data
        ports:
        - containerPort: 3000
      volumes:
      - name: feature-monitor-storage
        persistentVolumeClaim:
          claimName: feature-monitor-pvc
      - name: metabase-db-storage
        persistentVolumeClaim:
          claimName: metabase-db-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: metabase
spec:
  type: LoadBalancer
  selector:
    app: metabase
  ports:
  - port: 3000
    targetPort: 3000
```

## Step 2: Deploy Metabase

Apply the configuration:

```bash
kubectl apply -f metabase-deployment.yaml
```

## Step 3: Copy SQLite Database

Wait for the Metabase pod to be ready:

```bash
kubectl get pods
```

Once the pod is running, copy your SQLite database:

```bash
kubectl cp ./results/feature_monitor.db default/[pod-name]:/data/feature_monitor.db
```
Replace `[pod-name]` with your actual Metabase pod name (e.g., `metabase-cd996644-dnsz6`).

## Step 4: Access Metabase

1. Get the Metabase service URL:
```bash
kubectl get svc metabase
```

2. Access Metabase through your browser at `http://localhost:3000` or the IP address shown in the service output.

3. Complete the initial setup in the Metabase web interface.

## Step 5: Configure Database Connection

1. In Metabase's web interface, go to Admin settings
2. Click on Databases → Add database
3. Select SQLite as the database type
4. Set the database path to `/data/feature_monitor.db`
5. Save and test the connection

## Backup Metabase Configuration (Optional)

To backup your Metabase configuration:

```bash
kubectl cp default/[pod-name]:/metabase-data/metabase.db/metabase.db.mv.db ./metabase.db.mv.db
```

## Cleanup

To remove the Metabase deployment:

```bash
kubectl delete -f metabase-deployment.yaml
```

This will remove all resources created by the deployment, including the PersistentVolumeClaims.

## Troubleshooting

- If pods are stuck in `ContainerCreating` state, check the pod events:
```bash
kubectl describe pod [pod-name]
```

- To check pod logs:
```bash
kubectl logs [pod-name]
```

- If you need to restart the deployment:
```bash
kubectl rollout restart deployment metabase
```

        