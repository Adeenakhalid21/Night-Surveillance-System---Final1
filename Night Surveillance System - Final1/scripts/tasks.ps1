<#
Helper PowerShell tasks to avoid path mistakes and run common operations.

Usage (from repo root):
  . "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\tasks.ps1"
  Register-LOL
  Start-LOLTrainingFull
  Start-COCOTraining5K

Assumes venv at \.venv and inner project folder at:
  "Night Surveillance System - Final1\Night Surveillance System - Final1"
#>

function Get-RepoRoot {
  param()
  return (Get-Location).Path
}

function Get-InnerPath {
  param()
  return "Night Surveillance System - Final1\Night Surveillance System - Final1"
}

function Get-VenvPython {
  param([string]$RepoRoot)
  $path = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path $path)) { throw "Venv python not found: $path" }
  return $path
}

function Register-LOL {
  param(
    [string]$Name = "LOL LowLight",
    [string]$FolderRel = "datasets\lol_dataset\our485\low"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "scripts\register_dataset_folder.py"
  $folder = Join-Path $inner $FolderRel
  & $py $script --name $Name --folder $folder
}

function Start-LOLTrainingFast {
  param(
    [string]$Dataset = "LOL LowLight",
    [string]$Weights = "yolov8s.pt",
    [string]$RunName = "lol_lowlight_fast",
    [string]$OutRel = "datasets\prepared\lol_lowlight_pseudo"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "train_pseudolabel_yolo.py"
  $weightsPath = Join-Path $root $Weights
  $outPath = Join-Path $inner $OutRel
  & $py $script `
    --dataset $Dataset `
    --weights $weightsPath `
    --epochs 10 --batch 8 --imgsz 512 --conf 0.30 `
    --val_split 0.15 --resume-labels --infer-batch 8 `
    --device cpu --label-every 2 --fast-preset --patience 3 --freeze 8 `
    --workers 0 --name $RunName `
    --out $outPath `
    --seed 17
}

function Start-LOLTrainingFull {
  param(
    [string]$Dataset = "LOL LowLight",
    [string]$Weights = "yolov8s.pt",
    [string]$RunName = "lol_lowlight_full",
    [string]$OutRel = "datasets\prepared\lol_lowlight_full_pseudo"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "train_pseudolabel_yolo.py"
  $weightsPath = Join-Path $root $Weights
  $outPath = Join-Path $inner $OutRel
  & $py $script `
    --dataset $Dataset `
    --weights $weightsPath `
    --epochs 25 --batch 16 --imgsz 640 --conf 0.28 `
    --val_split 0.15 --resume-labels --infer-batch 8 `
    --device cpu --label-every 0 --cos-lr --patience 5 --freeze 4 `
    --workers 0 --name $RunName `
    --out $outPath `
    --seed 17
}

function Register-COCO-Full {
  param(
    [string]$Name = "COCO Train2017 Full",
    [string]$FolderRel = "datasets\train2017"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "scripts\register_dataset_folder.py"
  $folder = Join-Path $inner $FolderRel
  & $py $script --name $Name --folder $folder
}

function Start-COCOTraining5K {
  param(
    [string]$Dataset = "COCO Train2017 Full",
    [string]$Weights = "yolov8s.pt",
    [string]$RunName = "coco_full_5k_fast",
    [string]$OutRel = "datasets\prepared\coco_full_5k_pseudo"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "train_pseudolabel_yolo.py"
  $weightsPath = Join-Path $root $Weights
  $outPath = Join-Path $inner $OutRel
  & $py $script `
    --dataset $Dataset `
    --weights $weightsPath `
    --epochs 15 --batch 16 --imgsz 640 --conf 0.35 `
    --val_split 0.1 --resume-labels --infer-batch 8 `
    --device cpu --label-every 3 --fast-preset --cos-lr --patience 3 --freeze 10 `
    --workers 0 --name $RunName `
    --max-images 5000 `
    --out $outPath `
    --seed 42
}

function Start-HandWeaponTraining {
  param(
    [string]$WeightsRel = "runs\detect\dataset2_weapon_detection_best.pt",
    [string]$RunName = "gun_knife_hand_ft",
    [string]$OutRel = "datasets\prepared\gun_knife_binary"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "scripts\train_gun_knife_yolo.py"
  $gunSource = Join-Path $inner "datasets\gun_dataser"
  $knifeSource = Join-Path $inner "datasets\prepared\coco_full5k_pseudo"
  $outPath = Join-Path $inner $OutRel

  $weightsPath = Join-Path $inner $WeightsRel
  if (-not (Test-Path $weightsPath)) {
    $weightsPath = "auto"
    Write-Host "[INFO] Base checkpoint not found at requested path. Using --weights auto fallback."
  }

  & $py $script `
    --gun-source $gunSource `
    --knife-source $knifeSource `
    --out $outPath `
    --weights $weightsPath `
    --epochs 20 --batch 12 --imgsz 512 `
    --device cpu --workers 0 --patience 6 --freeze 6 `
    --name $RunName
}

function Use-HandWeaponWeights {
  param([string]$RunName = "gun_knife_hand_ft")
  $env:YOLO_WEIGHTS = "runs\detect\$RunName\weights\best.pt"
  $env:IMPORTANT_CLASSES = "person,knife,gun,backpack"
  $env:ANOMALY_CLASSES = "person,car,motorcycle,truck,backpack,knife,gun"
  Write-Host "Set YOLO_WEIGHTS to $env:YOLO_WEIGHTS"
  Write-Host "Set IMPORTANT_CLASSES to $env:IMPORTANT_CLASSES"
}

function Use-App-Weights {
  param([string]$RunName)
  $env:YOLO_WEIGHTS = "runs\detect\$RunName\weights\best.pt"
  $env:DISABLE_DETECTION_DB = "1"
  Write-Host "Set YOLO_WEIGHTS to $env:YOLO_WEIGHTS"
}

function Start-ObjectPersonWeaponTraining {
  param(
    [string]$RunName = "object_person_weapon_ft",
    [int]$Epochs = 20,
    [string]$Weights = "auto"
  )
  $root = Get-RepoRoot
  $inner = Get-InnerPath
  $py = Get-VenvPython -RepoRoot $root
  $script = Join-Path $inner "scripts\train_object_person_weapon_model.py"
  $personSource = Join-Path $inner "datasets\prepared\coco_full5k_pseudo"
  $weaponSource = Join-Path $inner "datasets\prepared\gun_knife_binary"
  $outPath = Join-Path $inner "datasets\prepared\object_person_weapon"

  & $py $script `
    --person-source $personSource `
    --weapon-source $weaponSource `
    --out $outPath `
    --weights $Weights `
    --epochs $Epochs --batch 10 --imgsz 512 `
    --device cpu --workers 0 --patience 6 --freeze 6 `
    --name $RunName
}

function Use-ObjectPersonWeaponWeights {
  param([string]$RunName = "object_person_weapon_ft")
  $env:YOLO_WEIGHTS = "runs\detect\$RunName\weights\best.pt"
  $env:IMPORTANT_CLASSES = "person,knife,gun,backpack,car,motorcycle,bus,truck"
  $env:ANOMALY_CLASSES = "knife,gun,weapon,intruder"
  $env:SAVE_DETECTION_FRAMES = "0"
  $env:SAVE_ANOMALY_FRAMES = "0"
  Write-Host "Set YOLO_WEIGHTS to $env:YOLO_WEIGHTS"
  Write-Host "Frame saving disabled for realtime speed (SAVE_DETECTION_FRAMES=0, SAVE_ANOMALY_FRAMES=0)."
}

Write-Host "Loaded tasks.ps1. Available commands: Register-LOL, Start-LOLTrainingFast, Start-LOLTrainingFull, Register-COCO-Full, Start-COCOTraining5K, Start-HandWeaponTraining, Use-HandWeaponWeights, Start-ObjectPersonWeaponTraining, Use-ObjectPersonWeaponWeights, Use-App-Weights"
