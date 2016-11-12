#!/bin/bash
kubectl --kubeconfig /home/xuerq/src/k8s/admin.conf delete rc tensorflow-ps-rc tensorflow-worker0-rc tensorflow-worker1-rc

