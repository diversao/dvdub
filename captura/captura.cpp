// captura.exe <pid>
// Grava so o som de um processo (e dos filhos dele) usando o "process loopback"
// do Windows 10 2004+ e escreve PCM 16 bits, 16 kHz, mono no stdout.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

// --- definicoes que o MinGW nao traz (audioclientactivationparams.h) ---
enum AUDIOCLIENT_ACTIVATION_TYPE_ { ACT_DEFAULT = 0, ACT_PROCESS_LOOPBACK = 1 };
enum PROCESS_LOOPBACK_MODE_ { PLM_INCLUDE_TREE = 0, PLM_EXCLUDE_TREE = 1 };
struct AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS_ { DWORD TargetProcessId; int ProcessLoopbackMode; };
struct AUDIOCLIENT_ACTIVATION_PARAMS_ { int ActivationType; AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS_ ProcessLoopbackParams; };

MIDL_INTERFACE("72A22D78-CDE4-431D-B8CC-843A71199B6D")
IActivateAudioInterfaceAsyncOperation_ : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE GetActivateResult(HRESULT *hr, IUnknown **iface) = 0;
};
MIDL_INTERFACE("41D949AB-9862-444A-80F6-C261334DA5EB")
IActivateAudioInterfaceCompletionHandler_ : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation_ *op) = 0;
};
DEFINE_GUID(IID_Completion_, 0x41D949AB, 0x9862, 0x444A, 0x80, 0xF6, 0xC2, 0x61, 0x33, 0x4D, 0xA5, 0xEB);
DEFINE_GUID(IID_IAgileObject_, 0x94ea2b94, 0xe9cc, 0x49e0, 0xc0, 0xff, 0xee, 0x64, 0xca, 0x8f, 0x5b, 0x90);
DEFINE_GUID(IID_IAudioClient_, 0x1CB9AD4C, 0xDBFA, 0x4c32, 0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2);
DEFINE_GUID(IID_IAudioCaptureClient_, 0xC8ADBD64, 0xE71E, 0x48a0, 0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17);

typedef HRESULT(WINAPI *ActivateFn)(LPCWSTR, REFIID, PROPVARIANT *, IActivateAudioInterfaceCompletionHandler_ *,
                                    IActivateAudioInterfaceAsyncOperation_ **);

struct Handler : IActivateAudioInterfaceCompletionHandler_ {
    LONG ref = 1;
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HRESULT hr = E_FAIL;
    IAudioClient *client = nullptr;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppv) override {
        if (riid == IID_IUnknown || riid == IID_Completion_ || riid == IID_IAgileObject_) {
            *ppv = this; AddRef(); return S_OK;
        }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref); }
    ULONG STDMETHODCALLTYPE Release() override { return InterlockedDecrement(&ref); }
    HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation_ *op) override {
        IUnknown *u = nullptr;
        HRESULT ahr = E_FAIL;
        op->GetActivateResult(&ahr, &u);
        hr = ahr;
        if (SUCCEEDED(ahr) && u) { u->QueryInterface(IID_IAudioClient_, (void **)&client); u->Release(); }
        SetEvent(done);
        return S_OK;
    }
};

static void die(const char *msg, HRESULT hr) { fprintf(stderr, "ERRO: %s (0x%08lx)\n", msg, (unsigned long)hr); exit(1); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: captura.exe <pid>\n"); return 2; }
    DWORD pid = (DWORD)strtoul(argv[1], nullptr, 10);
    _setmode(_fileno(stdout), _O_BINARY);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    HMODULE mm = LoadLibraryW(L"Mmdevapi.dll");
    ActivateFn activate = mm ? (ActivateFn)GetProcAddress(mm, "ActivateAudioInterfaceAsync") : nullptr;
    if (!activate) die("ActivateAudioInterfaceAsync indisponivel", E_FAIL);

    AUDIOCLIENT_ACTIVATION_PARAMS_ ap = {};
    ap.ActivationType = ACT_PROCESS_LOOPBACK;
    ap.ProcessLoopbackParams.TargetProcessId = pid;
    ap.ProcessLoopbackParams.ProcessLoopbackMode = PLM_INCLUDE_TREE;
    PROPVARIANT pv = {};
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(ap);
    pv.blob.pBlobData = (BYTE *)&ap;

    Handler *h = new Handler();
    IActivateAudioInterfaceAsyncOperation_ *op = nullptr;
    HRESULT hr = activate(L"VAD\\Process_Loopback", IID_IAudioClient_, &pv, h, &op);
    if (FAILED(hr)) die("ativar captura", hr);
    WaitForSingleObject(h->done, INFINITE);
    if (FAILED(h->hr) || !h->client) die("ativacao recusada", h->hr);
    IAudioClient *client = h->client;

    WAVEFORMATEX wf = {};
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = 1;
    wf.nSamplesPerSec = 16000;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = 2;
    wf.nAvgBytesPerSec = 32000;

    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                                AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                            200000, 0, &wf, nullptr);
    if (FAILED(hr)) die("Initialize", hr);
    client->SetEventHandle(ev);
    IAudioCaptureClient *cap = nullptr;
    hr = client->GetService(IID_IAudioCaptureClient_, (void **)&cap);
    if (FAILED(hr)) die("GetService", hr);
    hr = client->Start();
    if (FAILED(hr)) die("Start", hr);
    fprintf(stderr, "OK capturando pid %lu\n", (unsigned long)pid);

    static short silence[16000];
    for (;;) {
        WaitForSingleObject(ev, 100);
        UINT32 n = 0;
        while (SUCCEEDED(cap->GetNextPacketSize(&n)) && n > 0) {
            BYTE *data; UINT32 frames; DWORD flags;
            if (FAILED(cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
            const void *src = (flags & AUDCLNT_BUFFERFLAGS_SILENT) ? (const void *)silence : data;
            size_t bytes = (size_t)frames * 2;
            if (fwrite(src, 1, bytes, stdout) != bytes) return 0;  // leitor fechou
            cap->ReleaseBuffer(frames);
        }
        fflush(stdout);
    }
}
