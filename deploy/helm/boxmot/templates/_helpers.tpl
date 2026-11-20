{{/* Never truncate identities: collisions can create a second owner of a camera. */}}
{{- define "boxmot.fullname" -}}
{{- $name := default .Release.Name .Values.fullnameOverride -}}
{{- if not (regexMatch "^[a-z0-9]([-a-z0-9]{0,28}[a-z0-9])?$" $name) -}}
{{- fail "release name/fullnameOverride must be a DNS label of at most 30 characters" -}}
{{- end -}}
{{- $name -}}
{{- end -}}

{{- define "boxmot.labels" -}}
app.kubernetes.io/name: boxmot
app.kubernetes.io/instance: {{ .root.Release.Name | quote }}
app.kubernetes.io/component: {{ .component | quote }}
app.kubernetes.io/managed-by: {{ .root.Release.Service | quote }}
app.kubernetes.io/version: {{ .root.Chart.AppVersion | quote }}
helm.sh/chart: {{ printf "%s-%s" .root.Chart.Name .root.Chart.Version | replace "+" "_" | quote }}
{{- with .camera }}
boxmot.io/camera: {{ . | quote }}
{{- end }}
{{- end -}}

{{- define "boxmot.selectorLabels" -}}
app.kubernetes.io/name: boxmot
app.kubernetes.io/instance: {{ .root.Release.Name | quote }}
app.kubernetes.io/component: {{ .component | quote }}
{{- with .camera }}
boxmot.io/camera: {{ . | quote }}
{{- end }}
{{- end -}}

{{- define "boxmot.serviceAccountName" -}}
{{- if .settings.serviceAccount.create -}}
{{- default .name .settings.serviceAccount.name -}}
{{- else -}}
{{- default "default" .settings.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* This block is indented by callers at the PodSpec level. */}}
{{- define "boxmot.podSettings" -}}
serviceAccountName: {{ include "boxmot.serviceAccountName" . | quote }}
automountServiceAccountToken: false
enableServiceLinks: false
{{- with .root.Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .settings.runtimeClassName }}
runtimeClassName: {{ . | quote }}
{{- end }}
{{- with .settings.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .settings.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .settings.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

{{- define "boxmot.databaseEnv" -}}
- name: BOXMOT_ATTRIBUTES_DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Values.database.existingSecret | quote }}
      key: {{ .Values.database.urlKey | quote }}
{{- end -}}

{{- define "boxmot.claimName" -}}
{{- default .name .persistence.existingClaim -}}
{{- end -}}

{{- define "boxmot.pvc" -}}
{{- if not .persistence.existingClaim }}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ .name }}
  labels:
    {{- include "boxmot.labels" . | nindent 4 }}
  {{- if .persistence.retain }}
  annotations:
    helm.sh/resource-policy: keep
  {{- end }}
spec:
  accessModes:
    {{- toYaml .persistence.accessModes | nindent 4 }}
  {{- if eq .persistence.storageClass "-" }}
  storageClassName: ""
  {{- else if .persistence.storageClass }}
  storageClassName: {{ .persistence.storageClass | quote }}
  {{- end }}
  resources:
    requests:
      storage: {{ .persistence.size | quote }}
{{- end }}
{{- end -}}

{{- define "boxmot.serviceAccount" -}}
{{- if .settings.serviceAccount.create }}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "boxmot.serviceAccountName" . }}
  labels:
    {{- include "boxmot.labels" . | nindent 4 }}
  {{- with .settings.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
automountServiceAccountToken: false
{{- end }}
{{- end -}}

{{- define "boxmot.streamConfig" -}}
{{- $config := deepCopy .settings.config -}}
{{- $_ := set $config "camera_id" .camera -}}
{{- $_ := set $config "source_url_env" "BOXMOT_CAMERA_URL" -}}
{{- $_ := set $config "output_dir" "/outputs" -}}
{{- $_ := set $config "health_port" 8080 -}}
{{- $endpoint := .root.Values.vllm.externalEndpoint -}}
{{- if .root.Values.vllm.enabled -}}
{{- $endpoint = printf "http://%s-vllm:8000" (include "boxmot.fullname" .root) -}}
{{- end -}}
{{- $_ := set $config "attributes_host" $endpoint -}}
{{- toYaml $config -}}
{{- end -}}
