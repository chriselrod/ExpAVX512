
This library has an example SIMD exp, exp2, and exp10 implementation for AVX512.
This library requires supports AVX512.
It currently only implements support for `float`.

Note that `exp2` is considerably more efficient at the moment, so all else equal, prefer that.
2^(x*log2(e))
Perhaps you can fold in a multiplication by $log2(e) \approx 1.442695$ in a manner that doesn't lose accuracy.

Or, perhaps you can come up with a way to compensate for the loss of precision in `x*log2(e)`.

These are sample benchmark results, obtained on an Intel i9 10980XE (Cascadelake, essentially Skylake-AVX512; note that this CPU still experiences downclocking when running AVX and AVX512 instructions, which later Intel and AMD CPUs do not):
```
-----------------------------------------------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations CACHE-MISSES     CYCLES INSTRUCTIONS
-----------------------------------------------------------------------------------------------------------------
BM_simd_exp_run<4, 1>/128                101 ns          101 ns      6842615       6.138u     471.25          571
BM_simd_exp_run<4, 1>/256                170 ns          169 ns      4131515     25.4144u    764.647       1.131k
BM_simd_exp_run<4, 1>/512                295 ns          290 ns      2361965     11.0078u   1.35419k       2.251k
BM_simd_exp_run<4, 1>/1024               610 ns          599 ns      1168752     29.9465u   2.78697k       4.491k
BM_simd_exp_run<4, 1>/2048              1157 ns         1137 ns       611027     3.27318u   5.35288k       8.971k
BM_simd_exp_run<4, 1>/4096              2282 ns         2243 ns       308400     12.9702u   10.5324k      17.931k
BM_simd_exp_run<4, 1>/8192              4537 ns         4459 ns       154710     187.447u   20.9138k      35.851k
BM_simd_exp_run<4, 1>/16384             9046 ns         8892 ns        78462     433.331u    41.668k      71.691k
BM_simd_exp_run<4, 1>/32768            18111 ns        17801 ns        38988     1.59023m   83.1701k     143.371k
BM_simd_exp_run<4, 1>/65536            36405 ns        35775 ns        19297     2.12468m   168.424k     286.731k
BM_simd_exp_run<4, 1>/131072           73696 ns        72321 ns         9465    0.0293714   337.509k     573.451k
BM_simd_exp_run<4, 1>/262144          150991 ns       147968 ns         4731      1.58782   686.825k     1.14689M
BM_simd_exp_run<4, 1>/524288          305367 ns       299332 ns         2286     0.814961   1.39665M     2.29377M
BM_simd_exp_run<4, 1>/1048576         608722 ns       596229 ns         1166      15.5935   2.77903M     4.58753M
BM_simd_exp_run<4, 2>/128               94.7 ns         93.0 ns      7599033     4.60585u     434.24          570
BM_simd_exp_run<4, 2>/256                153 ns          150 ns      4667901     3.21344u    703.477        1.13k
BM_simd_exp_run<4, 2>/512                272 ns          268 ns      2617736     2.67407u   1.25361k        2.25k
BM_simd_exp_run<4, 2>/1024               563 ns          555 ns      1267139     17.3619u    2.5979k        4.49k
BM_simd_exp_run<4, 2>/2048              1076 ns         1073 ns       632306     83.8202u    5.0242k        8.97k
BM_simd_exp_run<4, 2>/4096              2127 ns         2121 ns       331384     63.3706u   9.86796k       17.93k
BM_simd_exp_run<4, 2>/8192              4168 ns         4157 ns       167369     35.8489u   19.5458k       35.85k
BM_simd_exp_run<4, 2>/16384             8322 ns         8300 ns        84826     259.354u   38.8187k       71.69k
BM_simd_exp_run<4, 2>/32768            16693 ns        16650 ns        41671    0.0357323   77.5501k      143.37k
BM_simd_exp_run<4, 2>/65536            33670 ns        33579 ns        20983       8.769m   156.457k      286.73k
BM_simd_exp_run<4, 2>/131072           68632 ns        68386 ns        10288    0.0146773   319.543k      573.45k
BM_simd_exp_run<4, 2>/262144          138004 ns       137394 ns         5145    0.0571429   638.699k     1.14689M
BM_simd_exp_run<4, 2>/524288          268805 ns       267591 ns         2611      0.20337   1.25077M     2.29377M
BM_simd_exp_run<4, 2>/1048576         550657 ns       547986 ns         1291      15.8947   2.55162M     4.58753M
BM_simd_exp_run<4, 4>/128               84.6 ns         84.3 ns      8154243     3.67907u    384.613          546
BM_simd_exp_run<4, 4>/256                132 ns          132 ns      4532082     1.10325u    617.129       1.082k
BM_simd_exp_run<4, 4>/512                271 ns          271 ns      3019497     2.64945u   1.08388k       2.154k
BM_simd_exp_run<4, 4>/1024               525 ns          524 ns      1000000          31u    2.0969k       4.298k
BM_simd_exp_run<4, 4>/2048               865 ns          863 ns       815373     11.0379u   4.05338k       8.586k
BM_simd_exp_run<4, 4>/4096              1715 ns         1710 ns       409926     80.5023u   7.94825k      17.162k
BM_simd_exp_run<4, 4>/8192              3341 ns         3333 ns       203966     29.4167u   15.7051k      34.314k
BM_simd_exp_run<4, 4>/16384             6785 ns         6767 ns       103471    0.0164007   31.2947k      68.618k
BM_simd_exp_run<4, 4>/32768            13309 ns        13277 ns        52950     1.35977m   62.4231k     137.226k
BM_simd_exp_run<4, 4>/65536            28760 ns        28256 ns        24857     9.37362m   131.023k     274.442k
BM_simd_exp_run<4, 4>/131072           56082 ns        54986 ns        12182     0.147841   255.548k     548.874k
BM_simd_exp_run<4, 4>/262144          116387 ns       115471 ns         6066      1.88328   522.225k     1.09774M
BM_simd_exp_run<4, 4>/524288          230410 ns       229408 ns         3044      1.99409   1.04952M     2.19547M
BM_simd_exp_run<4, 4>/1048576         463845 ns       461764 ns         1526      36.1606   2.13018M     4.39092M
BM_simd_exp_run<8, 1>/128               82.7 ns         82.1 ns      8386194     23.2525u    322.225          291
BM_simd_exp_run<8, 1>/256                126 ns          126 ns      5643281     2.65803u    488.924          571
BM_simd_exp_run<8, 1>/512                206 ns          205 ns      3290703     14.8904u    798.741       1.131k
BM_simd_exp_run<8, 1>/1024               402 ns          401 ns      1728617     53.8002u   1.57046k       2.251k
BM_simd_exp_run<8, 1>/2048               755 ns          753 ns       904214     36.4958u   2.94912k       4.491k
BM_simd_exp_run<8, 1>/4096              1475 ns         1471 ns       479318     227.406u   5.73544k       8.971k
BM_simd_exp_run<8, 1>/8192              2925 ns         2917 ns       237706     445.929u   11.5207k      17.931k
BM_simd_exp_run<8, 1>/16384             5842 ns         5826 ns       119177     360.808u   22.8293k      35.851k
BM_simd_exp_run<8, 1>/32768            11721 ns        11687 ns        60904     295.547u   45.5232k      71.691k
BM_simd_exp_run<8, 1>/65536            23695 ns        23631 ns        29847     837.605u   92.5272k     143.371k
BM_simd_exp_run<8, 1>/131072           51882 ns        51701 ns        13422     0.295485   201.273k     286.731k
BM_simd_exp_run<8, 1>/262144          103389 ns       102991 ns         6675     0.228464   398.021k     573.451k
BM_simd_exp_run<8, 1>/524288          210292 ns       209345 ns         3413      1.57867   825.843k     1.14689M
BM_simd_exp_run<8, 1>/1048576         451150 ns       448936 ns         1582       50.426   1.75483M     2.29377M
BM_simd_exp_run<8, 2>/128               78.1 ns         77.9 ns      8720182     8.60074u    309.117          290
BM_simd_exp_run<8, 2>/256                121 ns          119 ns      5809544     2.58196u    470.656          570
BM_simd_exp_run<8, 2>/512                191 ns          191 ns      3664864     24.8304u    759.643        1.13k
BM_simd_exp_run<8, 2>/1024               372 ns          371 ns      1881972     2.65679u   1.48123k        2.25k
BM_simd_exp_run<8, 2>/2048               702 ns          700 ns      1008921     146.691u   2.77443k        4.49k
BM_simd_exp_run<8, 2>/4096              1362 ns         1358 ns       522282     292.945u   5.37633k        8.97k
BM_simd_exp_run<8, 2>/8192              2711 ns         2703 ns       259460     281.354u   10.5614k       17.93k
BM_simd_exp_run<8, 2>/16384             5344 ns         5330 ns       130472     137.961u   20.9294k       35.85k
BM_simd_exp_run<8, 2>/32768            10608 ns        10580 ns        63997     234.386u    41.677k       71.69k
BM_simd_exp_run<8, 2>/65536            21068 ns        21010 ns        33282     2.13329m   83.4299k      143.37k
BM_simd_exp_run<8, 2>/131072           42632 ns        42468 ns        16332      5.8168m     170.6k      286.73k
BM_simd_exp_run<8, 2>/262144           99100 ns        98687 ns         7071    0.0504879   391.235k      573.45k
BM_simd_exp_run<8, 2>/524288          193061 ns       192277 ns         3584     0.341239     767.5k     1.14689M
BM_simd_exp_run<8, 2>/1048576         405535 ns       397029 ns         1786      36.5521   1.55886M     2.29377M
BM_simd_exp_run<8, 4>/128               71.1 ns         69.9 ns     10084253     3.47076u    277.693          278
BM_simd_exp_run<8, 4>/256                105 ns          104 ns      6795412     16.6289u    406.405          546
BM_simd_exp_run<8, 4>/512                169 ns          166 ns      4186879     11.9421u    654.726       1.082k
BM_simd_exp_run<8, 4>/1024               308 ns          303 ns      2358921     23.3157u   1.16844k       2.154k
BM_simd_exp_run<8, 4>/2048               564 ns          554 ns      1255645     70.8799u   2.16818k       4.298k
BM_simd_exp_run<8, 4>/4096              1076 ns         1057 ns       649894     72.3195u    4.2051k       8.586k
BM_simd_exp_run<8, 4>/8192              2106 ns         2080 ns       333605     194.841u   8.28173k      17.162k
BM_simd_exp_run<8, 4>/16384             4133 ns         4121 ns       170120     405.596u   16.3841k      34.314k
BM_simd_exp_run<8, 4>/32768             8165 ns         8142 ns        83936     250.191u   32.5139k      68.618k
BM_simd_exp_run<8, 4>/65536            16536 ns        16490 ns        42801     3.38777m   65.7706k     137.226k
BM_simd_exp_run<8, 4>/131072           34973 ns        34842 ns        19942    0.0866011   138.122k     274.442k
BM_simd_exp_run<8, 4>/262144           84259 ns        83904 ns         8331     0.175129   323.253k     548.874k
BM_simd_exp_run<8, 4>/524288          170070 ns       169364 ns         4177     0.609528   656.954k     1.09774M
BM_simd_exp_run<8, 4>/1048576         352872 ns       350997 ns         1991      118.426   1.38386M     2.19547M
BM_simd_exp_run<16, 1>/128              63.6 ns         63.1 ns     11202497     2.14238u    250.209          160
BM_simd_exp_run<16, 1>/256              85.2 ns         83.7 ns      8333738     1.91991u    332.174          312
BM_simd_exp_run<16, 1>/512               130 ns          129 ns      5475616     6.39197u     512.92          616
BM_simd_exp_run<16, 1>/1024              234 ns          233 ns      2986746     2.00888u     927.38       1.224k
BM_simd_exp_run<16, 1>/2048              420 ns          419 ns      1675759     10.7414u   1.66267k        2.44k
BM_simd_exp_run<16, 1>/4096              783 ns          781 ns       885101     109.592u   3.04656k       4.872k
BM_simd_exp_run<16, 1>/8192             1800 ns         1795 ns       369378     221.995u   7.12268k       9.736k
BM_simd_exp_run<16, 1>/16384            3661 ns         3650 ns       189671     453.417u   14.1684k      19.464k
BM_simd_exp_run<16, 1>/32768            7201 ns         7181 ns        95812     313.113u   28.1683k       38.92k
BM_simd_exp_run<16, 1>/65536           14712 ns        14672 ns        47237     2.60389m   57.2477k      77.832k
BM_simd_exp_run<16, 1>/131072          31692 ns        31582 ns        21973    0.0226187   123.573k     155.656k
BM_simd_exp_run<16, 1>/262144          80904 ns        80579 ns         8790     0.228783    312.28k     311.304k
BM_simd_exp_run<16, 1>/524288         162723 ns       162098 ns         4306     0.998607   628.659k       622.6k
BM_simd_exp_run<16, 1>/1048576        331166 ns       329628 ns         2136      67.0197    1.2704M     1.24519M
BM_simd_exp_run<16, 2>/128              62.9 ns         62.7 ns     11122896      3.8659u    243.357          153
BM_simd_exp_run<16, 2>/256              81.4 ns         81.1 ns      8581207     21.9083u    315.208          293
BM_simd_exp_run<16, 2>/512               123 ns          122 ns      5636531      887.07n    473.947          573
BM_simd_exp_run<16, 2>/1024              211 ns          211 ns      3353178     3.57869u    818.799       1.133k
BM_simd_exp_run<16, 2>/2048              374 ns          373 ns      1888826     25.4126u   1.44929k       2.253k
BM_simd_exp_run<16, 2>/4096              702 ns          700 ns       984040            0     2.725k       4.493k
BM_simd_exp_run<16, 2>/8192             1473 ns         1468 ns       476837     10.4858u   5.73827k       8.973k
BM_simd_exp_run<16, 2>/16384            2908 ns         2900 ns       237544     4.20975u    11.291k      17.933k
BM_simd_exp_run<16, 2>/32768            5740 ns         5725 ns       120373     8.30751u    22.469k      35.853k
BM_simd_exp_run<16, 2>/65536           11994 ns        11960 ns        57370     296.322u   46.7923k      71.693k
BM_simd_exp_run<16, 2>/131072          27323 ns        27230 ns        25460     4.90966m   106.755k     143.373k
BM_simd_exp_run<16, 2>/262144          78416 ns        78117 ns         8967     0.359429   301.964k     286.733k
BM_simd_exp_run<16, 2>/524288         158412 ns       157795 ns         4455     0.731089   606.875k     573.453k
BM_simd_exp_run<16, 2>/1048576        319621 ns       318216 ns         2205      52.6553   1.23362M     1.14689M
BM_simd_exp_run<16, 4>/128              58.4 ns         58.3 ns     11895571     1.26097u    226.972          147
BM_simd_exp_run<16, 4>/256              75.1 ns         74.9 ns      9355737     748.204n    294.388          281
BM_simd_exp_run<16, 4>/512               108 ns          108 ns      6434818     310.809n    424.221          549
BM_simd_exp_run<16, 4>/1024              180 ns          180 ns      3837621     1.56347u    703.301       1.085k
BM_simd_exp_run<16, 4>/2048              319 ns          318 ns      2166203     11.5409u   1.24221k       2.157k
BM_simd_exp_run<16, 4>/4096              597 ns          596 ns      1191770     11.7472u   2.33025k       4.301k
BM_simd_exp_run<16, 4>/8192             1220 ns         1216 ns       582008     6.87276u   4.75915k       8.589k
BM_simd_exp_run<16, 4>/16384            2364 ns         2358 ns       295611     47.3595u   9.33597k      17.165k
BM_simd_exp_run<16, 4>/32768            4784 ns         4770 ns       147537     142.337u   18.5228k      34.317k
BM_simd_exp_run<16, 4>/65536            9930 ns         9901 ns        71263     954.212u    38.663k      68.621k
BM_simd_exp_run<16, 4>/131072          22578 ns        22504 ns        31223    0.0443904    88.049k     137.229k
BM_simd_exp_run<16, 4>/262144          77503 ns        77196 ns         9110    0.0917673   299.796k     274.445k
BM_simd_exp_run<16, 4>/524288         155874 ns       155252 ns         4511      1.04478   599.646k     548.877k
BM_simd_exp_run<16, 4>/1048576        313546 ns       312176 ns         2248       7.6855   1.20591M     1.09774M
BM_simd_exp_run<4, 1, 2>/128            45.7 ns         45.6 ns     15372594     390.305n     211.65          475
BM_simd_exp_run<4, 1, 2>/256            96.2 ns         95.9 ns      7279447     16.3474u    440.017          939
BM_simd_exp_run<4, 1, 2>/512             190 ns          189 ns      3747724     2.66829u    868.823       1.867k
BM_simd_exp_run<4, 1, 2>/1024            432 ns          431 ns      1639872     1.21961u   1.97998k       3.723k
BM_simd_exp_run<4, 1, 2>/2048            855 ns          853 ns       833398     1.19991u   3.92348k       7.435k
BM_simd_exp_run<4, 1, 2>/4096           1708 ns         1704 ns       410284      29.248u   7.87678k      14.859k
BM_simd_exp_run<4, 1, 2>/8192           3464 ns         3455 ns       203349     536.024u   15.8041k      29.707k
BM_simd_exp_run<4, 1, 2>/16384          6920 ns         6903 ns       100087     479.583u   31.5238k      59.403k
BM_simd_exp_run<4, 1, 2>/32768         13675 ns        13642 ns        50998     372.564u   63.1191k     118.795k
BM_simd_exp_run<4, 1, 2>/65536         27908 ns        27832 ns        25300     4.38735m   127.304k     237.579k
BM_simd_exp_run<4, 1, 2>/131072        59208 ns        59017 ns        11908     0.111354   268.078k     475.147k
BM_simd_exp_run<4, 1, 2>/262144       113443 ns       113016 ns         6218     0.373271   514.947k     950.283k
BM_simd_exp_run<4, 1, 2>/524288       230293 ns       229433 ns         3072        1.389   1.04155M     1.90056M
BM_simd_exp_run<4, 1, 2>/1048576      481763 ns       479708 ns         1467      46.9127   2.16409M      3.8011M
BM_simd_exp_run<4, 2, 2>/128            43.7 ns         43.5 ns     16135203     929.644n    196.941          474
BM_simd_exp_run<4, 2, 2>/256            87.7 ns         87.4 ns      8131256     17.3405u    393.674          938
BM_simd_exp_run<4, 2, 2>/512             174 ns          173 ns      4106753     16.0711u    790.473       1.866k
BM_simd_exp_run<4, 2, 2>/1024            396 ns          395 ns      1817244     86.9448u   1.77767k       3.722k
BM_simd_exp_run<4, 2, 2>/2048            778 ns          776 ns       911274     131.684u   3.53264k       7.434k
BM_simd_exp_run<4, 2, 2>/4096           1534 ns         1520 ns       462754     1.00485m   6.98961k      14.858k
BM_simd_exp_run<4, 2, 2>/8192           3072 ns         3064 ns       226478     1.77942m   14.2051k      29.706k
BM_simd_exp_run<4, 2, 2>/16384          6310 ns         6253 ns       114623     1.68378m   28.3927k      59.402k
BM_simd_exp_run<4, 2, 2>/32768         12344 ns        12303 ns        55458    0.0167514   56.6832k     118.794k
BM_simd_exp_run<4, 2, 2>/65536         24929 ns        24857 ns        27757    0.0105199   115.089k     237.578k
BM_simd_exp_run<4, 2, 2>/131072        52014 ns        51836 ns        13863    0.0626848   235.588k     475.146k
BM_simd_exp_run<4, 2, 2>/262144       102648 ns       102271 ns         6847    0.0452753   467.016k     950.282k
BM_simd_exp_run<4, 2, 2>/524288       207424 ns       206626 ns         3413      1.01904   938.475k     1.90055M
BM_simd_exp_run<4, 2, 2>/1048576      424958 ns       423366 ns         1613      239.133   1.88394M      3.8011M
BM_simd_exp_run<4, 4, 2>/128            39.3 ns         39.2 ns     17845328     5.54767u    178.602          450
BM_simd_exp_run<4, 4, 2>/256            78.6 ns         78.4 ns      9070283     11.5763u    357.765          890
BM_simd_exp_run<4, 4, 2>/512             156 ns          156 ns      4446127     899.659n    717.863        1.77k
BM_simd_exp_run<4, 4, 2>/1024            335 ns          334 ns      2104534     2.85099u   1.49258k        3.53k
BM_simd_exp_run<4, 4, 2>/2048            686 ns          684 ns      1008900     6.93825u   3.10244k        7.05k
BM_simd_exp_run<4, 4, 2>/4096           1378 ns         1374 ns       516918     245.687u   6.19019k       14.09k
BM_simd_exp_run<4, 4, 2>/8192           2696 ns         2689 ns       256725     11.6857u   12.3724k       28.17k
BM_simd_exp_run<4, 4, 2>/16384          5441 ns         5427 ns       128562     54.4484u   24.6721k       56.33k
BM_simd_exp_run<4, 4, 2>/32768         10765 ns        10738 ns        65743     15.2107u   49.2549k      112.65k
BM_simd_exp_run<4, 4, 2>/65536         21799 ns        21736 ns        32358     8.25144m   98.8428k      225.29k
BM_simd_exp_run<4, 4, 2>/131072        44172 ns        44018 ns        15951    0.0384302   200.628k      450.57k
BM_simd_exp_run<4, 4, 2>/262144        95730 ns        95364 ns         7309     0.114653    433.23k      901.13k
BM_simd_exp_run<4, 4, 2>/524288       192679 ns       191915 ns         3709      1.33082   862.543k     1.80225M
BM_simd_exp_run<4, 4, 2>/1048576      385876 ns       384214 ns         1830      63.0399   1.72514M     3.60449M
BM_simd_exp_run<8, 1, 2>/128            29.8 ns         29.7 ns     23709286     3.07896u     114.19          243
BM_simd_exp_run<8, 1, 2>/256            58.5 ns         58.3 ns     11848717     15.0227u    224.872          475
BM_simd_exp_run<8, 1, 2>/512             124 ns          122 ns      5660293     42.5773u    470.493          939
BM_simd_exp_run<8, 1, 2>/1024            274 ns          272 ns      2642464     94.2302u   1.06698k       1.867k
BM_simd_exp_run<8, 1, 2>/2048            541 ns          531 ns      1319367     21.2223u   2.10412k       3.723k
BM_simd_exp_run<8, 1, 2>/4096           1098 ns         1079 ns       645020     103.873u   4.22446k       7.435k
BM_simd_exp_run<8, 1, 2>/8192           2267 ns         2260 ns       314756     2.87207m   8.72167k      14.859k
BM_simd_exp_run<8, 1, 2>/16384          4498 ns         4480 ns       150624     9.08886m   17.3452k      29.707k
BM_simd_exp_run<8, 1, 2>/32768          9079 ns         8921 ns        75802    0.0394449   34.8358k      59.403k
BM_simd_exp_run<8, 1, 2>/65536         18014 ns        17740 ns        38694    0.0214245   70.5027k     118.795k
BM_simd_exp_run<8, 1, 2>/131072        37760 ns        37025 ns        18931     7.28963m   147.957k     237.579k
BM_simd_exp_run<8, 1, 2>/262144        93320 ns        91416 ns         7722     0.123155   363.525k     475.147k
BM_simd_exp_run<8, 1, 2>/524288       189044 ns       185196 ns         3804     0.299685   731.758k     950.284k
BM_simd_exp_run<8, 1, 2>/1048576      384647 ns       376508 ns         1817      40.6957   1.48483M     1.90056M
BM_simd_exp_run<8, 2, 2>/128            27.7 ns         27.2 ns     25873867     3.55571u    108.603          242
BM_simd_exp_run<8, 2, 2>/256            52.9 ns         52.2 ns     13665976     7.82966u    205.518          474
BM_simd_exp_run<8, 2, 2>/512             108 ns          108 ns      6348955     31.0287u    415.827          938
BM_simd_exp_run<8, 2, 2>/1024            245 ns          243 ns      2929543     62.4671u    957.748       1.866k
BM_simd_exp_run<8, 2, 2>/2048            493 ns          484 ns      1440370     88.8661u   1.88981k       3.722k
BM_simd_exp_run<8, 2, 2>/4096            991 ns          973 ns       712362     92.6495u   3.81446k       7.434k
BM_simd_exp_run<8, 2, 2>/8192           2027 ns         1991 ns       351851     216.001u   7.84716k      14.858k
BM_simd_exp_run<8, 2, 2>/16384          4015 ns         4004 ns       172124     395.064u    15.749k      29.706k
BM_simd_exp_run<8, 2, 2>/32768          7939 ns         7916 ns        88700     1.14994m   31.4611k      59.402k
BM_simd_exp_run<8, 2, 2>/65536         16183 ns        15912 ns        43231     5.96794m   62.6816k     118.794k
BM_simd_exp_run<8, 2, 2>/131072        34447 ns        34097 ns        21021      1.47077   133.248k     237.578k
BM_simd_exp_run<8, 2, 2>/262144        84583 ns        84233 ns         7665     0.638878   327.039k     475.146k
BM_simd_exp_run<8, 2, 2>/524288       179550 ns       177192 ns         3983      0.57419   689.523k     950.282k
BM_simd_exp_run<8, 2, 2>/1048576      345493 ns       343834 ns         1951      16.3911    1.3321M     1.90055M
BM_simd_exp_run<8, 4, 2>/128            28.7 ns         28.6 ns     24105744     2.94536u    111.092          230
BM_simd_exp_run<8, 4, 2>/256            49.0 ns         48.9 ns     13873951     6.12659u    189.506          450
BM_simd_exp_run<8, 4, 2>/512             100 ns         99.8 ns      7042937     567.945n      381.8          890
BM_simd_exp_run<8, 4, 2>/1024            207 ns          206 ns      3397339     44.4466u     785.18        1.77k
BM_simd_exp_run<8, 4, 2>/2048            399 ns          398 ns      1743355     1.72082u   1.55715k        3.53k
BM_simd_exp_run<8, 4, 2>/4096            806 ns          804 ns       871059     3.44408u   3.15473k        7.05k
BM_simd_exp_run<8, 4, 2>/8192           1650 ns         1645 ns       436800     199.176u   6.36097k       14.09k
BM_simd_exp_run<8, 4, 2>/16384          3342 ns         3332 ns       213608     791.169u   12.7082k       28.17k
BM_simd_exp_run<8, 4, 2>/32768          6631 ns         6611 ns       106432     450.992u   25.3881k       56.33k
BM_simd_exp_run<8, 4, 2>/65536         13124 ns        13082 ns        53164      7.3546m   51.3158k      112.65k
BM_simd_exp_run<8, 4, 2>/131072        29777 ns        29654 ns        23150    0.0109719   117.212k      225.29k
BM_simd_exp_run<8, 4, 2>/262144        81641 ns        81269 ns         8564    0.0367819   323.096k      450.57k
BM_simd_exp_run<8, 4, 2>/524288       164046 ns       163308 ns         4269     0.277348   650.824k      901.13k
BM_simd_exp_run<8, 4, 2>/1048576      328410 ns       326806 ns         2147      11.2473   1.30313M     1.80225M
BM_simd_exp_run<16, 1, 2>/128           16.0 ns         15.9 ns     44600224       627.8n    63.2223          134
BM_simd_exp_run<16, 1, 2>/256           29.3 ns         29.2 ns     23299640     8.15463u    114.554          262
BM_simd_exp_run<16, 1, 2>/512           65.3 ns         65.1 ns     11320391     25.3525u    247.789          518
BM_simd_exp_run<16, 1, 2>/1024           139 ns          139 ns      4854883     10.7109u    547.607        1.03k
BM_simd_exp_run<16, 1, 2>/2048           281 ns          280 ns      2510128     40.6354u   1.08696k       2.054k
BM_simd_exp_run<16, 1, 2>/4096           558 ns          557 ns      1226551     1.63059u   2.18541k       4.102k
BM_simd_exp_run<16, 1, 2>/8192          1377 ns         1373 ns       509880     49.0311u   5.35157k       8.198k
BM_simd_exp_run<16, 1, 2>/16384         2769 ns         2761 ns       256111     331.887u   10.6964k       16.39k
BM_simd_exp_run<16, 1, 2>/32768         6365 ns         6272 ns       107630    0.0203568   24.6493k      32.774k
BM_simd_exp_run<16, 1, 2>/65536        11735 ns        11600 ns        62068    0.0763517   45.2171k      65.542k
BM_simd_exp_run<16, 1, 2>/131072       26277 ns        26194 ns        26848     4.76758m   102.177k     131.078k
BM_simd_exp_run<16, 1, 2>/262144       77662 ns        77363 ns         9010     0.334739   299.845k      262.15k
BM_simd_exp_run<16, 1, 2>/524288      157369 ns       156760 ns         4451     0.560099   606.199k     524.294k
BM_simd_exp_run<16, 1, 2>/1048576     300813 ns       297529 ns         2224      45.8719   1.15376M     1.04858M
BM_simd_exp_run<16, 2, 2>/128           16.1 ns         15.8 ns     44394584     2.07232u    62.7249          129
BM_simd_exp_run<16, 2, 2>/256           28.3 ns         27.8 ns     25062974     3.19196u    108.503          245
BM_simd_exp_run<16, 2, 2>/512           56.7 ns         56.0 ns     12663007     7.10732u     216.32          477
BM_simd_exp_run<16, 2, 2>/1024           120 ns          119 ns      5898343     37.1291u    460.186          941
BM_simd_exp_run<16, 2, 2>/2048           240 ns          239 ns      2924456     101.899u     918.72       1.869k
BM_simd_exp_run<16, 2, 2>/4096           499 ns          497 ns      1424139     111.646u   1.89902k       3.725k
BM_simd_exp_run<16, 2, 2>/8192          1178 ns         1174 ns       594421     146.361u   4.50889k       7.437k
BM_simd_exp_run<16, 2, 2>/16384         2329 ns         2321 ns       300434     4.35703m   8.96247k      14.861k
BM_simd_exp_run<16, 2, 2>/32768         4695 ns         4680 ns       149530     555.073u   17.9555k      29.709k
BM_simd_exp_run<16, 2, 2>/65536         9412 ns         9383 ns        73425     0.142145   36.2299k      59.405k
BM_simd_exp_run<16, 2, 2>/131072       22341 ns        22251 ns        31736     0.086999   84.6989k     118.797k
BM_simd_exp_run<16, 2, 2>/262144       77772 ns        77471 ns         8969     0.156428   298.395k     237.581k
BM_simd_exp_run<16, 2, 2>/524288      156712 ns       156059 ns         4498       1.4731   598.642k     475.149k
BM_simd_exp_run<16, 2, 2>/1048576     312543 ns       311209 ns         2238      23.2051   1.21998M     950.285k
BM_simd_exp_run<16, 4, 2>/128           15.3 ns         15.2 ns     45963847     87.0249n    59.5339          123
BM_simd_exp_run<16, 4, 2>/256           25.7 ns         25.6 ns     27032023     36.9932n    99.9058          233
BM_simd_exp_run<16, 4, 2>/512           52.3 ns         52.1 ns     13372896     149.556n    200.577          453
BM_simd_exp_run<16, 4, 2>/1024           109 ns          109 ns      6443652     620.766n    424.463          893
BM_simd_exp_run<16, 4, 2>/2048           218 ns          218 ns      3204169      936.28n    843.409       1.773k
BM_simd_exp_run<16, 4, 2>/4096           435 ns          434 ns      1620470     2.46842u   1.68415k       3.533k
BM_simd_exp_run<16, 4, 2>/8192           932 ns          929 ns       755122     115.213u   3.59165k       7.053k
BM_simd_exp_run<16, 4, 2>/16384         1867 ns         1862 ns       361749     69.1087u   7.16785k      14.093k
BM_simd_exp_run<16, 4, 2>/32768         3744 ns         3732 ns       186731     380.226u   14.3383k      28.173k
BM_simd_exp_run<16, 4, 2>/65536         7601 ns         7581 ns        92008     336.927u   29.7274k      56.333k
BM_simd_exp_run<16, 4, 2>/131072       19898 ns        19836 ns        35677     2.04614m   77.2103k     112.653k
BM_simd_exp_run<16, 4, 2>/262144       77364 ns        77064 ns         9077    0.0208219   299.152k     225.293k
BM_simd_exp_run<16, 4, 2>/524288      156707 ns       156104 ns         4466     0.326243   602.699k     450.573k
BM_simd_exp_run<16, 4, 2>/1048576     322397 ns       321058 ns         2180      7.89771   1.24357M     901.133k
```
As sizes increase, we become memory bottlenecked. For smaller arrays, e.g. of length 1024, larger vectors significantly improve throughput.
While there is no dependency between iterations, `exp` kernel has a large dependency chain. A CPU should be able to speculate across loop iterations, but in practice this CPU benefits, e.g. going from no unrolling in `exp<16,1>` to unrolling by `exp<16,4>` with 1024-element vectors, we get a roughly 15% improvement in IPC (instructions per clock) on top of a roughly 12% decease in the number of instructions required.
`exp2`, on top of needing significantly fewer instructions, achieved much better instructions per clock cycle on this computer.
