{{- define "boxmot.maintenancePod" -}}
metadata:
  labels:
    {{- include "boxmot.selectorLabels" . | nindent 4 }}
    {{- with .settings.podLabels }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  annotations:
    boxmot.io/secret-revision: {{ .root.Values.secretRevision | quote }}
    {{- with .settings.podAnnotations }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
spec:
  {{- include "boxmot.podSettings" . | nindent 2 }}
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    fsGroup: 10001
    fsGroupChangePolicy: OnRootMismatch
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: maintenance
      image: {{ .root.Values.image | quote }}
      imagePullPolicy: {{ .root.Values.imagePullPolicy }}
      command:
        {{- toYaml .command | nindent 8 }}
      securityContext:
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
      env:
        {{- include "boxmot.databaseEnv" .root | nindent 8 }}
      resources:
        {{- toYaml .settings.resources | nindent 8 }}
      {{- if .outputClaim }}
      volumeMounts:
        - {name: outputs, mountPath: /outputs}
      {{- end }}
  {{- if .outputClaim }}
  volumes:
    - name: outputs
      persistentVolumeClaim:
        claimName: {{ .outputClaim }}
  {{- end }}
{{- end -}}
