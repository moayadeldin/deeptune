; Inno Setup script for the DeepTune desktop app installer.
;
; Packages the PyInstaller "onedir" build (packaging\dist\DeepTune\, produced
; by packaging\build_exe.ps1 or deeptune.spec) into a single Windows
; installer EXE: DeepTune-Setup-<version>.exe. That installer is the one
; file end users download -- it copies the already-frozen app (no Python
; required on their machine) into the current user's profile (no admin
; rights needed -- see PrivilegesRequired/DefaultDirName below), adds Start
; Menu / optional Desktop shortcuts, and registers an uninstaller.
;
; Build with packaging\build_installer.ps1 (runs the PyInstaller build
; first, then this), or directly once packaging\dist\DeepTune exists:
;   ISCC packaging\installer.iss /DMyAppVersion=1.1.0
;
; MyAppVersion defaults below if not passed with /D (e.g. run from the IDE).
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

#define MyAppName "DeepTune"
#define MyAppPublisher "DeepTune"
#define MyAppExeName "DeepTune.exe"
#define MyAppURL "https://github.com/moayadeldin/deeptune"
#define SourceDir "dist\DeepTune"

[Setup]
AppId={{6C6C6E6A-6465-4570-9454-756E652D3131}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Onedir PyInstaller builds are large (the ML stack alone is a couple of GB);
; lzma2/ultra64 keeps the download smaller at the cost of a slower build.
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=DeepTune-Setup-{#MyAppVersion}
OutputDir=installer-dist
SetupIconFile=assets\deeptune.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
; No admin rights required: installs under the current user's Local AppData
; instead of Program Files, since end users downloading this off a release
; page won't reliably have (or want to grant) admin rights.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove any PyInstaller onedir temp/cache files DeepTune itself may have
; written under {app} at runtime (results/session data live elsewhere, in
; the user's chosen output folder and %LOCALAPPDATA%\DeepTune, so they are
; deliberately left alone by an uninstall).
Type: filesandordirs; Name: "{app}\__pycache__"
