#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
import torch.nn.functional as F  # Dodaj ten import, jeśli go brakuje na górze
import os

##########################################################

##########################################################

args_strModel = "sintel-final"
args_strOne = "./images/one.png"
args_strTwo = "./images/two.png"
args_strOut = "./out.flo"

# Parsowanie argumentów powinno być w bloku if __name__ == '__main__',
# aby nie było wykonywane przy imporcie modułu.
# Na razie zakomentuję, bo ustawiamy args_strModel z zewnątrz.
# for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
#     'model=',
#     'one=',
#     'two=',
#     'out=',
# ])[0]:
#     if strOption == '--model' and strArg != '': args_strModel = strArg
#     if strOption == '--one' and strArg != '': args_strOne = strArg
#     if strOption == '--two' and strArg != '': args_strTwo = strArg
#     if strOption == '--out' and strArg != '': args_strOut = strArg
# # end

##########################################################

backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    grid_key = str(tenFlow.shape)  # Użyj jako klucza
    if (
        grid_key not in backwarp_tenGrid
        or backwarp_tenGrid[grid_key].device != tenFlow.device
    ):
        tenHor = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
            .view(1, 1, 1, -1)
            .repeat(1, 1, tenFlow.shape[2], 1)
        )
        tenVer = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, tenFlow.shape[3])
        )
        backwarp_tenGrid[grid_key] = torch.cat([tenHor, tenVer], 1)

    # Normalizacja przepływu
    norm_flow_x = tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0))
    norm_flow_y = tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    tenFlow_normalized = torch.cat([norm_flow_x, norm_flow_y], 1)

    return F.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[grid_key] + tenFlow_normalized).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


##########################################################

# --- Przenieś definicje klas Preprocess i Basic na poziom modułu ---


class Preprocess(
    torch.nn.Module
):  # Zmieniono nazwę, aby uniknąć konfliktu z `Network.Preprocess`
    def __init__(self):
        super().__init__()

    # end

    def forward(self, tenInput):
        # Upewnij się, że tensory mean/std są na tym samym urządzeniu co tenInput
        mean_val = torch.tensor(
            data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device
        ).view(1, 3, 1, 1)
        std_val = torch.tensor(
            data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
            dtype=tenInput.dtype,
            device=tenInput.device,
        ).view(1, 3, 1, 1)

        tenInput = tenInput.flip([1])  # BGR conversion
        tenInput = tenInput - mean_val
        tenInput = tenInput * std_val
        return tenInput

    # end


# end


class Basic(torch.nn.Module):  # Zmieniono nazwę
    def __init__(
        self, intLevel
    ):  # intLevel może nie być tu potrzebny, jeśli architektura jest stała
        super().__init__()
        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3
            ),
        )

    # end

    def forward(self, tenInput):
        return self.netBasic(tenInput)

    # end


# end


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Teraz używamy klas zdefiniowanych na poziomie modułu
        self.netPreprocess = Preprocess()  # Użyj `Preprocess` zdefiniowanego wyżej
        self.netBasic = torch.nn.ModuleList(
            [Basic(intLevel) for intLevel in range(6)]
        )  # Użyj `Basic`

        # Ładowanie wag - upewnij się, że args_strModel jest ustawione PRZED tą inicjalizacją
        # To jest obsługiwane w FlowDataset.__init__
        try:
            url = (
                "http://content.sniklaus.com/github/pytorch-spynet/network-"
                + args_strModel
                + ".pytorch"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url=url, file_name="spynet-" + args_strModel
            )
            self.load_state_dict(
                {
                    strKey.replace("module", "net"): tenWeight
                    for strKey, tenWeight in state_dict.items()
                }
            )
        except Exception as e:
            print(
                f"BŁĄD podczas ładowania wag dla oryginalnego SPyNet (model: {args_strModel}): {e}"
            )
            print(
                "Upewnij się, że masz połączenie z internetem i nazwa modelu jest poprawna."
            )
            print("Oryginalny SPyNet może nie działać poprawnie bez wag.")
            # Możesz rzucić błędem, jeśli wagi są krytyczne: raise e

    # end

    def forward(self, tenOne, tenTwo):
        # tenFlow = [] # Nie jest używane

        tenOne_pyramid = [self.netPreprocess(tenOne)]
        tenTwo_pyramid = [self.netPreprocess(tenTwo)]

        for intLevel in range(
            5
        ):  # Tworzy 6 poziomów piramidy (0 to najdrobniejszy po preprocessingu)
            if tenOne_pyramid[0].shape[2] > 32 or tenOne_pyramid[0].shape[3] > 32:
                tenOne_pyramid.insert(
                    0,
                    F.avg_pool2d(
                        input=tenOne_pyramid[0],
                        kernel_size=2,
                        stride=2,
                        count_include_pad=False,
                    ),
                )
                tenTwo_pyramid.insert(
                    0,
                    F.avg_pool2d(
                        input=tenTwo_pyramid[0],
                        kernel_size=2,
                        stride=2,
                        count_include_pad=False,
                    ),
                )
            else:  # Jeśli obraz jest już za mały, nie zmniejszaj dalej
                break
        # Po tej pętli, tenOne_pyramid[0] i tenTwo_pyramid[0] to najgrubsze poziomy piramidy

        # Inicjalizacja przepływu na najgrubszym poziomie
        coarsest_level_one = tenOne_pyramid[0]
        initial_flow_h = int(math.floor(coarsest_level_one.shape[2] / 2.0))
        initial_flow_w = int(math.floor(coarsest_level_one.shape[3] / 2.0))

        # Upewnij się, że wymiary nie są zerowe lub ujemne
        if initial_flow_h <= 0 or initial_flow_w <= 0:
            # To może się zdarzyć, jeśli obrazy wejściowe są bardzo małe,
            # a piramida redukuje je do zbyt małych wymiarów.
            # Dla 192x320 to nie powinno być problemem.
            # 192 / (2^5) = 6; 320 / (2^5) = 10. Więc /2 daje 3x5.
            print(
                f"OSTRZEŻENIE: Wymiary początkowego przepływu są bardzo małe lub zerowe: {initial_flow_h}x{initial_flow_w}"
            )
            print(f"Wymiary najgrubszego poziomu: {coarsest_level_one.shape}")
            # Można rzucić błąd lub spróbować ustawić minimalny rozmiar
            initial_flow_h = max(1, initial_flow_h)
            initial_flow_w = max(1, initial_flow_w)

        current_flow = coarsest_level_one.new_zeros(
            [coarsest_level_one.shape[0], 2, initial_flow_h, initial_flow_w]
        )

        # Iteracja od najgrubszego do najdrobniejszego poziomu piramidy
        # len(tenOne_pyramid) to liczba poziomów piramidy
        for intLevel_idx in range(
            len(tenOne_pyramid)
        ):  # intLevel_idx idzie od 0 (najgrubszy) do N-1 (najdrobniejszy)
            tenUpsampled = (
                F.interpolate(
                    input=current_flow,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                * 2.0
            )

            # Obrazy na bieżącym poziomie piramidy (zaczynając od najgrubszego)
            # Pyramida jest zbudowana tak, że indeks 0 to najgrubszy poziom.
            # Pętla `self.netBasic` w oryginalnym kodzie była `range(len(tenOne))`
            # gdzie `tenOne` (lista piramidy) była budowana od najdrobniejszego do najgrubszego (insert(0, ...))
            # a potem iteracja była po tej liście.
            # `self.netBasic[intLevel]` - tutaj `intLevel` powinien odpowiadać poziomowi piramidy.
            # Jeśli `self.netBasic` ma 6 modułów, a `len(tenOne_pyramid)` też jest 6, to pasuje.

            current_level_one = tenOne_pyramid[intLevel_idx]
            current_level_two = tenTwo_pyramid[intLevel_idx]

            # Dopasuj rozmiar upsamplowanego przepływu do bieżącego poziomu piramidy
            if tenUpsampled.shape[2] != current_level_one.shape[2]:
                tenUpsampled = F.pad(
                    input=tenUpsampled,
                    pad=[0, 0, 0, current_level_one.shape[2] - tenUpsampled.shape[2]],
                    mode="replicate",
                )
            if tenUpsampled.shape[3] != current_level_one.shape[3]:
                tenUpsampled = F.pad(
                    input=tenUpsampled,
                    pad=[0, current_level_one.shape[3] - tenUpsampled.shape[3], 0, 0],
                    mode="replicate",
                )

            # Wejście do modułu Basic na tym poziomie
            concat_input = torch.cat(
                [
                    current_level_one,
                    backwarp(tenInput=current_level_two, tenFlow=tenUpsampled),
                    tenUpsampled,
                ],
                1,
            )

            # Użyj modułu Basic odpowiadającego temu poziomowi piramidy.
            # Oryginalny kod używał `self.netBasic[intLevel]`, gdzie `intLevel` rosło.
            # Musimy się upewnić, że indeksacja `self.netBasic` jest spójna z iteracją po piramidzie.
            # Jeśli `self.netBasic` ma 6 modułów, a `len(tenOne_pyramid)` jest 6 (poziomy 0 do 5),
            # to `self.netBasic[intLevel_idx]` powinno być OK.
            current_flow = self.netBasic[intLevel_idx](concat_input) + tenUpsampled
        # end

        return current_flow  # Zwraca przepływ na najdrobniejszym poziomie

    # end


# end

netNetwork = None  # Globalna instancja używana przez estimate

##########################################################


def estimate(
    tenOne, tenTwo
):  # Ta funkcja jest bardziej dla CLI, my używamy Network.forward
    global netNetwork
    global args_strModel  # Potrzebne, jeśli netNetwork jest None i musi być utworzony

    if netNetwork is None:
        # args_strModel musi być ustawione przed wywołaniem Network()
        netNetwork = (
            Network().cuda().train(False)
        )  # Network() użyje globalnej args_strModel
    # end

    # Reszta funkcji estimate jest specyficzna dla CLI i interpolacji do oryginalnego rozmiaru
    # My będziemy wywoływać Network.forward bezpośrednio w Dataset na zaaugmentowanych obrazach
    # o stałym rozmiarze (np. 192x320), więc nie potrzebujemy tej logiki interpolacji tutaj.

    # Dla spójności, jeśli chcesz użyć `estimate`, musisz zapewnić, że `tenOne` i `tenTwo`
    # są tensorami [3, H, W] w zakresie [0,1]
    # oraz że `args_strModel` jest ustawione.

    # Poniższa część jest z oryginalnego estimate, ale może nie być potrzebna w naszym kontekście Datasetu
    # if tenOne.shape[0] != 1 or tenOne.shape[1] !=3 : # estimate oczekuje B=1, C=3
    #     tenOne = tenOne.unsqueeze(0)
    # if tenTwo.shape[0] != 1 or tenTwo.shape[1] !=3 :
    #     tenTwo = tenTwo.unsqueeze(0)

    # device = next(netNetwork.parameters()).device
    # tenOne = tenOne.to(device)
    # tenTwo = tenTwo.to(device)

    # return netNetwork(tenOne, tenTwo).squeeze(0).cpu() # Prostsze wywołanie forward

    # --- ORYGINALNA LOGIKA estimate ---
    assert tenOne.dim() == 3 and tenTwo.dim() == 3  # Oczekuje HWC * 1/255.0
    assert tenOne.shape[0] == 3 and tenTwo.shape[0] == 3  # C, H, W
    assert tenOne.shape[1] == tenTwo.shape[1] and tenOne.shape[2] == tenTwo.shape[2]

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # Oryginalny skrypt miał tu asercje na rozmiar, np. 1024x436.
    # assert(intWidth == 1024)
    # assert(intHeight == 436)

    tenPreprocessedOne = tenOne.cuda().view(
        1, 3, intHeight, intWidth
    )  # Dodaj batch dim
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    # Skalowanie do wielokrotności 32
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = F.interpolate(
        input=tenPreprocessedOne,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )
    tenPreprocessedTwo = F.interpolate(
        input=tenPreprocessedTwo,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )

    # Wywołanie sieci
    tenFlow = netNetwork(tenPreprocessedOne, tenPreprocessedTwo)  # Wynik na GPU

    # Interpolacja wyniku z powrotem do oryginalnego rozmiaru
    tenFlow = F.interpolate(
        input=tenFlow, size=(intHeight, intWidth), mode="bilinear", align_corners=False
    )

    # Skalowanie komponentów przepływu
    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()  # Zwróć bez wymiaru batcha, na CPU


# end

##########################################################

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True

    # Ta część jest dla CLI, my nie będziemy jej używać bezpośrednio z datasetu.
    # Upewnij się, że argumenty CLI są parsowane, jeśli uruchamiasz ten skrypt bezpośrednio.
    # Tutaj zakładamy, że args_strModel, args_strOne, itp. są ustawione (np. przez getopt).
    # getopt.getopt powinien być wywołany tutaj, jeśli skrypt jest uruchamiany.

    # Przykładowe użycie:
    # Ustaw args_strModel, args_strOne, args_strTwo, args_strOut przed wywołaniem.
    # Np. z linii komend, lub hardkoduj do testów:
    # args_strModel = 'sintel-final'
    # args_strOne = './images/one.png' # Musisz mieć te obrazy
    # args_strTwo = './images/two.png'
    # args_strOut = './out.flo'

    # Sprawdź, czy pliki istnieją przed próbą otwarcia
    if os.path.exists(args_strOne) and os.path.exists(args_strTwo):
        tenOne = torch.FloatTensor(
            numpy.ascontiguousarray(
                numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1]
                .transpose(2, 0, 1)
                .astype(numpy.float32)
                * (1.0 / 255.0)
            )
        )
        tenTwo = torch.FloatTensor(
            numpy.ascontiguousarray(
                numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1]
                .transpose(2, 0, 1)
                .astype(numpy.float32)
                * (1.0 / 255.0)
            )
        )

        tenOutput = estimate(
            tenOne, tenTwo
        )  # estimate użyje globalnej netNetwork i args_strModel

        objOutput = open(args_strOut, "wb")
        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(
            objOutput
        )  # .flo magic number
        numpy.array([tenOutput.shape[2], tenOutput.shape[1]], numpy.int32).tofile(
            objOutput
        )  # width, height
        numpy.array(
            tenOutput.numpy(force=True).transpose(1, 2, 0), numpy.float32
        ).tofile(objOutput)  # data
        objOutput.close()
        print(f"Przepływ zapisany do {args_strOut}")
    else:
        print(
            f"BŁĄD: Nie znaleziono plików wejściowych: {args_strOne} lub {args_strTwo}"
        )
        print(
            "Uruchom z --one <plik1> --two <plik2> [--model <nazwa_modelu>] [--out <plik_wyjsciowy.flo>]"
        )
