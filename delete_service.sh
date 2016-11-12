#!/bin/bash
kubectl --kubeconfig /home/xuerq/src/k8s/admin.conf delete service tensorflow-ps-service tensorflow-wk-service0 tensorflow-wk-service1
