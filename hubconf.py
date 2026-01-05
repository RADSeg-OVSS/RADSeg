dependencies = ['torch', 'torchvision', 'numpy', 'timm', 'PIL', 'segment_anything']

def radseg_encoder(model_version="c-radio_v3-b", lang_model="siglip2", scra_scaling=10.0, scga_scaling=10.0, **kwargs):
    """
    Loads the RADSeg Encoder.
    Arguments:
        model_version (str): RADIO backbone version.
        lang_model (str): language adaptor to use.
        scra_scaling (float): SCRA scaling factor.
        scga_scaling (float): SCGA scaling factor.
    """
    from radseg.radseg import RADSegEncoder
    return RADSegEncoder(
        model_version=model_version,
        lang_model=lang_model,
        scra_scaling=scra_scaling,
        scga_scaling=scga_scaling,
        **kwargs
    )